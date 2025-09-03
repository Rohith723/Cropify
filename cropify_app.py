import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import pickle
import shutil
import kagglehub

# ------------------- Paths -------------------
DATA_DIR = "data"
MODEL_FILE = "crop_model.pkl"
SCALER_FILE = "scaler.pkl"

# ------------------- Step 1: Download dataset -------------------
if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    st.info("Downloading latest dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("madhuraatmarambhagat/crop-recommendation-dataset")
    st.success(f"Dataset downloaded at {dataset_path}")

    # Find CSV inside downloaded dataset
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        st.error("No CSV file found in Kaggle dataset!")
        st.stop()

    # Copy CSV to local data folder
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    shutil.copy(os.path.join(dataset_path, csv_files[0]), os.path.join(DATA_DIR, csv_files[0]))
    st.success(f"CSV file copied to {DATA_DIR}/{csv_files[0]}")

# ------------------- Step 2: Load CSV -------------------
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not csv_files:
    st.error("No CSV file found in data folder!")
    st.stop()

csv_path = os.path.join(DATA_DIR, csv_files[0])
df = pd.read_csv(csv_path)
st.success(f"Dataset loaded: {csv_files[0]} ({df.shape[0]} rows, {df.shape[1]} columns)")

# ------------------- Step 3: Normalize Columns -------------------
df.columns = df.columns.str.strip().str.lower()  # lowercase & strip spaces
TARGET_COL = "label"  # update to match dataset

if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' not found in dataset columns: {df.columns.tolist()}")
    st.stop()

# ------------------- Step 4: Train model or load pickle -------------------
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    st.success("Loaded trained model from disk.")
else:
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    st.success("Model trained and saved successfully!")

# ------------------- Step 5: Streamlit UI -------------------
st.title("🌾 Cropify - Crop Recommendation System")
st.write("Enter your farm parameters below:")

# Dynamically create input fields from dataset features
feature_cols = df.drop(TARGET_COL, axis=1).columns.tolist()
input_values = []

# Split features into two columns with default value 0
col1, col2 = st.columns(2)
for i, col_name in enumerate(feature_cols):
    default_val = 0.0  # default value set to 0
    if i % 2 == 0:
        val = col1.number_input(f"{col_name.capitalize()}", value=default_val)
    else:
        val = col2.number_input(f"{col_name.capitalize()}", value=default_val)
    input_values.append(val)

# Predict button
if st.button("Predict Crop"):
    input_data = pd.DataFrame([input_values], columns=feature_cols)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"✅ Recommended Crop: {prediction[0]}")
