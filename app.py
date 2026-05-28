import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import requests

api_key = os.getenv("API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
UPLOAD_FOLDER = "static/uploads"

model = load_model("efficientnet_crop_disease_model.keras", compile=False)
mandi_model = joblib.load("mandi_price_model.pkl")
mandi_dataset = pd.read_csv("Agriculture_price_dataset.csv")
mandi_dataset["Price Date"] = pd.to_datetime(mandi_dataset["Price Date"])
print("Mandi model and dataset loaded")

with open("disease_data.json") as f:
    disease_data = json.load(f)

with open("class_names.json") as f:
    class_names = json.load(f)


def weather_alert(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"

    try:
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return "Weather data unavailable for this location"
        desc = response["weather"][0]["description"].lower()
        if "rain" in desc or "drizzle" in desc or "thunderstorm" in desc:
            return "⚠ Heavy rain expected. Harvest early."
        elif "cloud" in desc:
            return "☁ Cloudy weather. Monitor crop conditions."
        elif "haze" in desc or "mist" in desc or "fog" in desc:
            return "🌫 Low visibility. Watch for fungal diseases."
        elif "clear" in desc:
            return "☀ Weather is clear. Good conditions."
        else:
            return "Weather conditions normal."
    except:
        return "Weather service error"

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = float(np.max(pred))
    predicted_class = class_names[idx]
    recommendation = disease_data.get(predicted_class, {})
    return predicted_class, confidence, recommendation

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(request: Request, leaf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, leaf.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await leaf.read())
    disease, confidence, info = predict_image(file_path)
    return templates.TemplateResponse(
    "index.html",
    {
        "request": request,
        "disease": disease,
        "confidence": round(confidence * 100, 2),
        "info": info,
        "filename": leaf.filename
    }
)

@app.post("/predict_mandi")
async def predict_mandi(request: Request):
    body = await request.json()
    crop = body["crop"]
    market = body["location"]
    weather_msg = weather_alert(market)
    production_cost = float(body["production_cost"])
    yield_quintals = float(body["yield"])
    distance = float(body["distance"])
    date = pd.to_datetime(body["date"])
    year = date.year
    month = date.month
    day = date.day

    filtered = mandi_dataset[
        (mandi_dataset["Commodity"] == crop) &
        (mandi_dataset["Market Name"] == market)
        ]
    
    if filtered.empty:
        return {"error": "No data available for this crop and market"}
    latest = filtered.sort_values("Price Date").iloc[-1]

    min_price = latest["Min_Price"]
    max_price = latest["Max_Price"]
    price_spread = max_price - min_price
    avg_price = (max_price + min_price) / 2
    input_data = pd.DataFrame(
        np.zeros((1, mandi_model.n_features_in_)),
        columns=mandi_model.feature_names_in_
    )

    input_data["year"] = year
    input_data["month"] = month
    input_data["day"] = day
    input_data["min_price"] = min_price
    input_data["max_price"] = max_price
    input_data["price_spread"] = price_spread
    input_data["avg_price"] = avg_price
    crop_col = "commodity_" + crop
    market_col = "market_" + market

    if crop_col in input_data.columns:
        input_data[crop_col] = 1

    if market_col in input_data.columns:
        input_data[market_col] = 1
    predicted_price = mandi_model.predict(input_data)[0]

    transport_cost = distance * 5
    net_profit = predicted_price - production_cost - transport_cost
    if predicted_price > production_cost * 1.10:
        recommendation = "HOLD"
    else:
        recommendation = "SELL"

    alerts = []
    alerts.append(weather_msg)

    if recommendation == "SELL":
        alerts.append("📉 Price predicted to drop. Sell now.")
        recommendation_text = "SELL (Price predicted to drop. Sell now.)"

    elif recommendation == "HOLD":
        alerts.append("📈 Prices may improve. You can hold.")
        recommendation_text = "HOLD (Prices may improve. You can hold.)"

    risk_level = "LOW"
    if "Rain" in weather_msg and recommendation == "SELL":
        risk_level = "HIGH"
    elif recommendation == "SELL":
        risk_level = "MEDIUM"
    elif "Rain" in weather_msg:
        risk_level = "MEDIUM"

    return {
        "best_mandi": market,
        "predicted_price": round(predicted_price, 2),
        "transport_cost": round(transport_cost, 2),
        "net_profit": round(net_profit, 2),
        "recommendation": recommendation_text,
        "risk_level": risk_level,
        "alerts": alerts
    }

@app.get("/mandi_options")
def mandi_options():
    crops = sorted(mandi_dataset["Commodity"].unique().tolist())
    return {
        "crops": crops
    }

@app.get("/markets_by_crop")
def markets_by_crop(crop: str):
    markets = mandi_dataset[
        mandi_dataset["Commodity"] == crop
    ]["Market Name"].unique().tolist()
    return {"markets": markets}