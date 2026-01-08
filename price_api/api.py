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
        df = predict_with_proxy(
            input_data=data.dict(),
            forecast_df=forecast_df,
            historic_df=historic_df
        )

        # Ensure datetime & sort
        df["month"] = pd.to_datetime(df["month"], form)
        df = df.sort_values("month")

        prediction_point_date = pd.to_datetime("2025-12-01")

        before_prediction_df = df[df["month"] < prediction_point_date]
        prediction_point_df = df[df["month"] == prediction_point_date]
        forecast_df_out = df[df["month"] > prediction_point_date]

        # Ensure SINGLE prediction point (important!)
        prediction_point_df = prediction_point_df.tail(1)

        # Format month (NO time)
        for _df in [before_prediction_df, prediction_point_df, forecast_df_out]:
            _df["month"] = _df["month"].dt.strftime("%d-%m-%Y")

        return {
            "before_prediction": before_prediction_df.to_dict(orient="records"),
            "prediction_point": prediction_point_df.to_dict(orient="records"),
            "forecast": forecast_df_out.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



