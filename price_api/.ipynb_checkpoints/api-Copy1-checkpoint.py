import os
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from predict_price import predict_with_proxy  # your model code

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_PATH = os.path.join(BASE_DIR, "forecast_df.csv")
HISTORIC_PATH = os.path.join(BASE_DIR, "historic_df.csv")

forecast_df = pd.read_csv(FORECAST_PATH)
historic_df = pd.read_csv(HISTORIC_PATH)

app = FastAPI(title="Price Prediction API", version="1.0")

class PredictionInput(BaseModel):
    area_name: str
    reg_type_en: str
    rooms_en: str
    land_type_en: str
    floor_bin: str
    developer_cat: str
    project_cat: str
    has_parking: int
    swimming_pool: int
    balcony: int
    elevator: int
    metro: int
    procedure_area: float

@app.get("/")
def home():
    return {"status": "API is running 🚀"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        df = predict_with_proxy(input_data=data.dict(),
                                forecast_df=forecast_df,
                                historic_df=historic_df)
        df["month"] = df["month"].dt.strftime("%Y-%m-%d")
        return {"rows": len(df), "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
