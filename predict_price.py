

import os
import pickle
import pandas as pd


# ============================================================
# 1. BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "FLIPOSE_DATA",
    "V_2.2",
    "Models_predictions"
)

print("Using Base Directory:\n", BASE_DIR)
print("\nFiles in directory:", os.listdir(BASE_DIR))


# ============================================================
# 2. DIRECT MODEL AREAS (NO PROXY MAPPING)
# ============================================================

DIRECT_MODEL_AREAS = {
    "Al Barsha South Fourth",
    "Business Bay",
    "Al Merkadh",
    "Burj Khalifa",
    "Hadaeq Sheikh Mohammed Bin Rashid",
    "Al Khairan First",
    "Wadi Al Safa 5",
    "Al Thanyah Fifth",
    "Al Barshaa South Third",
    "Jabal Ali First",
    "Madinat Al Mataar",
    "Madinat Dubai Almelaheyah",
    "Me'Aisem First",
    "Al Hebiah Fourth",
    "Al Barsha South Fifth",
    "Al Hebiah First",
    "Nadd Hessa",
    "Palm Jumeirah",
    "Al Barshaa South Second",
    "Al Yelayiss 2",
    "Al Warsan First"
}


# ============================================================
# 3. PROXY / GROUP MAPPING
# ============================================================

PROXY_MAPPING = {

    # ---------------- proxy1 ----------------
    "Al Barsha First": "proxy1",
    "Al Hebiah Second": "proxy1",
    "Al Hebiah Sixth": "proxy1",
    "Al Hebiah Third": "proxy1",
    "Madinat Hind 4": "proxy1",
    "Wadi Al Safa 3": "proxy1",
    "Wadi Al Safa 4": "proxy1",
    "Wadi Al Safa 7": "proxy1",

    # ---------------- proxy2 ----------------
    "Bukadra": "proxy2",
    "Hadaeq Sheikh Mohammed Bin Rashid": "proxy2",
    "Ras Al Khor Industrial First": "proxy2",
    "Jumeirah First": "proxy2",
    "Palm Deira": "proxy2",
    "Al Khairan First": "proxy2",

    # ---------------- proxy3 ----------------
    "Al Thanyah Third": "proxy3",
    "Jabal Ali Industrial Second": "proxy3",

    # ---------------- groups ----------------
    "Al Kifaf": "G1",
    "Warsan Fourth": "G3",
    "Jabal Ali": "G3",
    "Zaabeel Second": "G4",
    "Zaabeel First": "G4"
}


# ============================================================
# 4. LOAD TRAIN COLUMNS
# ============================================================
COL_DIR = os.path.join(BASE_DIR,
"Trained Columns"
)
def load_columns(model_key):
    for f in os.listdir(COL_DIR):
        if f.lower() == f"trained_columns_{model_key}.pkl".lower():
            with open(os.path.join(COL_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"❌ trained_columns_{model_key}.pkl not found")


# ============================================================
# 5. LOAD MODEL
# ============================================================
MODEL_DIR = os.path.join(BASE_DIR,
"Models"
)
def load_model(model_key):
    for f in os.listdir(MODEL_DIR):
        if f.lower() == f"rf_model_{model_key}.pkl".lower():
            with open(os.path.join(MODEL_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"❌ rf_model_{model_key}.pkl not found")


# ============================================================
# 6. MAIN PREDICTION FUNCTION
# ============================================================

def predict_with_proxy(input_data, forecast_df, historic_df):

    area = input_data["area_name"]

    # --------------------------------------------------------
    # Decide model key
    # --------------------------------------------------------
    if area in DIRECT_MODEL_AREAS:
        model_key = area
        model_type = "direct"
    elif area in PROXY_MAPPING:
        model_key = PROXY_MAPPING[area]
        model_type = "proxy"
    else:
        raise ValueError(f"❌ No model found for area: {area}")

    # --------------------------------------------------------
    # Load model & training columns
    # --------------------------------------------------------
    train_columns = load_columns(model_key)
    model = load_model(model_key)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------
    categorical_features = [
        "reg_type_en",
        "rooms_en",
        "land_type_en",
        "floor_bin",
        "developer_cat",
        "project_cat"
    ]

    binary_features = [
        "has_parking",
        "swimming_pool",
        "balcony",
        "elevator",
        "metro"
    ]

    continuous_features = ["procedure_area"]

    temp = pd.DataFrame([input_data])
    temp = temp[categorical_features + binary_features + continuous_features]

    temp = pd.get_dummies(
        temp,
        columns=categorical_features,
        drop_first=False
    )

    for col in train_columns:
        if col not in temp.columns:
            temp[col] = 0

    temp = temp[train_columns]

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    predicted_price = model.predict(temp)[0]
    print("Raw Model Prediction:", predicted_price)

    # --------------------------------------------------------
    # DIRECT MODEL → RETURN SINGLE VALUE
    # --------------------------------------------------------
    if model_type == "direct":
        return pd.DataFrame({
            "area": [area],
            "predicted_price": [predicted_price]
        })

    # --------------------------------------------------------
    # PROXY MODEL → APPLY FORECAST
    # --------------------------------------------------------
    #proxy_num = int(model_key.replace("proxy", ""))

    gf = forecast_df[forecast_df["area"] == model_key].copy()
    gf["median_price"] = predicted_price * gf["growth_factor"]

    historic = historic_df[historic_df["area"] == model_key].copy()
    if not historic.empty:
        historic.loc[historic.index[-1], "median_price"] = predicted_price

    final_df = pd.concat([historic, gf])[["month", "median_price", "area"]]
    final_df["month"] = pd.to_datetime(final_df["month"], format="%d-%m-%Y")
    
    final_df = (
        final_df
        .sort_values("month")
        .reset_index(drop=True)
    )


    return final_df


# ============================================================
# 7. EXAMPLE RUN
# ============================================================

if __name__ == "__main__":

    forecast_df = pd.read_csv("forecast_df.csv")
    historic_df = pd.read_csv("historic_df.csv")

    input_data = {
        "area_name": "Madinat Hind 4",

        # categorical
        "reg_type_en": "off-plan",
        "rooms_en": "1B/R",
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "developer_cat": "Grade 1",
        "project_cat": "Mid_Rise",

        # binary
        "has_parking": 1,
        "swimming_pool": 0,
        "balcony": 0,
        "elevator": 1,
        "metro": 1,

        # continuous
        "procedure_area": 70
    }

    final_output = predict_with_proxy(
        input_data,
        forecast_df,
        historic_df
    )

    print("\nFINAL OUTPUT:\n", final_output.tail(15))