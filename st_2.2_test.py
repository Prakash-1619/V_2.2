import streamlit as st
import pandas as pd
import os
import pickle
import plotly.express as px
import ast

# ============================================================
# GitHub auto-push setup (Step 2 + 5)
# ============================================================
github_user = st.secrets["github"]["username"]
github_token = st.secrets["github"]["token"]
repo_name = st.secrets["github"]["repo"]
branch = st.secrets["github"]["branch"]

# Remote URL with token (HTTPS)
repo_url = f"https://{github_token}@github.com/{github_user}/{repo_name}.git"

# Ensure Git remote is set correctly
os.system("git remote remove origin 2>/dev/null")  # remove old remote if exists
os.system(f"git remote add origin {repo_url}")
os.system(f"git checkout -B {branch}")  # ensure branch exists

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dubai Property Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏙️ Dubai Property Price Prediction Dashboard")

# ============================================================
# BASE DIRECTORY
# ============================================================
BASE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "FLIPOSE_DATA",
    "V_2.2",
    "Models_predictions"
)

MODEL_DIR = os.path.join(BASE_DIR, "Models")
COL_DIR = os.path.join(BASE_DIR, "Trained Columns")
INPUT_CSV = os.path.join(BASE_DIR, "V_2.2_inputs.csv")

# ============================================================
# LOAD INPUT FEATURE CSV
# ============================================================
feature_df = pd.read_csv(INPUT_CSV)

def parse_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except:
        return []

def get_area_row(area):
    row = feature_df[feature_df["area_name_en"] == area]
    return row.iloc[0] if not row.empty else None

# ============================================================
# DIRECT & PROXY CONFIG
# ============================================================
DIRECT_MODEL_AREAS = { ... }  # same as your existing code
PROXY_MAPPING = { ... }       # same as your existing code
ALL_AREAS = sorted(feature_df["area_name_en"].unique())

# ============================================================
# LOADERS
# ============================================================
def load_columns(model_key):
    for f in os.listdir(COL_DIR):
        if f.lower() == f"trained_columns_{model_key}.pkl".lower():
            with open(os.path.join(COL_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"trained_columns_{model_key}.pkl not found")

def load_model(model_key):
    for f in os.listdir(MODEL_DIR):
        if f.lower() == f"rf_model_{model_key}.pkl".lower():
            with open(os.path.join(MODEL_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"rf_model_{model_key}.pkl not found")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_with_proxy(input_data, forecast_df, historic_df):
    area = input_data["area_name"]
    if area in DIRECT_MODEL_AREAS:
        model_key = area
    elif area in PROXY_MAPPING:
        model_key = PROXY_MAPPING[area]
    else:
        raise ValueError("No model available")

    train_columns = load_columns(model_key)
    model = load_model(model_key)

    categorical_features = [
        "reg_type_en", "rooms_en", "land_type_en",
        "floor_bin", "developer_cat", "project_cat"
    ]

    binary_features = [
        "has_parking", "swimming_pool",
        "balcony", "elevator", "metro"
    ]

    continuous_features = ["procedure_area"]

    temp = pd.DataFrame([input_data])
    temp = temp[categorical_features + binary_features + continuous_features]
    temp = pd.get_dummies(temp, columns=categorical_features)

    for col in train_columns:
        if col not in temp.columns:
            temp[col] = 0

    temp = temp[train_columns]
    predicted_price = model.predict(temp)[0]

    gf = forecast_df[forecast_df["area"] == model_key].copy()
    hist = historic_df[historic_df["area"] == model_key].copy()

    if gf.empty and hist.empty:
        return pd.DataFrame({
            "month": ["Current"],
            "median_price": [predicted_price],
            "area": [area]
        })

    gf["median_price"] = predicted_price * gf["growth_factor"]
    hist.loc[hist.index[-1], "median_price"] = predicted_price

    final_df = pd.concat([hist, gf])
    final_df["month"] = pd.to_datetime(final_df["month"], format="%d-%m-%Y")
    final_df["area"] = area
    return final_df.sort_values("month")

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("📌 Property Inputs")

area = st.sidebar.selectbox("Area", ALL_AREAS)
row = get_area_row(area)

reg_type = st.sidebar.selectbox("Registration Type", parse_list(row["reg_type_en"]))
rooms = st.sidebar.selectbox("Rooms", parse_list(row["rooms_en"]))
land_type = st.sidebar.selectbox("Land Type", parse_list(row["land_type_en"]))
floor_bin = st.sidebar.selectbox("Floor Range", parse_list(row["floor_bin"]))
developer_cat = st.sidebar.selectbox("Developer Grade", parse_list(row["developer_cat"]))
project_cat = st.sidebar.selectbox("Project Type", parse_list(row["project_cat"]))

procedure_area = st.sidebar.number_input(
    "Area (sqm)",
    float(row["procedure_area_min"]),
    float(row["procedure_area_max"]),
    float(row["procedure_area_median"])
)

def binary_checkbox(label, values):
    return st.sidebar.checkbox(label) if 1 in parse_list(values) else False

has_parking = binary_checkbox("Parking", row["has_parking"])
swimming_pool = binary_checkbox("Swimming Pool", row["swimming_pool"])
balcony = binary_checkbox("Balcony", row["balcony"])
elevator = binary_checkbox("Elevator", row["elevator"])
metro = binary_checkbox("Near Metro", row["metro"])

# ============================================================
# RUN PREDICTION + AUTO SAVE + GIT PUSH
# ============================================================
if st.sidebar.button("🔮 Predict Price"):

    forecast_df = pd.read_csv("forecast_df.csv")
    historic_df = pd.read_csv("historic_df.csv")

    input_data = {
        "area_name": area,
        "reg_type_en": reg_type,
        "rooms_en": rooms,
        "land_type_en": land_type,
        "floor_bin": floor_bin,
        "developer_cat": developer_cat,
        "project_cat": project_cat,
        "has_parking": int(has_parking),
        "swimming_pool": int(swimming_pool),
        "balcony": int(balcony),
        "elevator": int(elevator),
        "metro": int(metro),
        "procedure_area": procedure_area
    }

    result = predict_with_proxy(input_data, forecast_df, historic_df)

    # --- Save prediction CSV ---
    save_file = f"latest_prediction_{area}.csv"
    result.to_csv(save_file, index=False)
    st.success(f"✅ Prediction saved to {save_file}")

    # --- Git add, commit, push ---
    os.system(f"git add {save_file}")
    os.system(f'git commit -m "Auto-update prediction for {area}"')
    os.system(f"git push origin {branch}")

    # --- Display chart & table ---
    st.subheader("📈 Price Trend")
    fig = px.line(
        result,
        x="month",
        y="median_price",
        title=f"Predicted Price Trend – {area}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Prediction Table")
    st.dataframe(result, use_container_width=True)
