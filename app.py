import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pune House Price Prediction",
    layout="centered"
)

st.title("🏠 Pune House Price Prediction")
st.caption("Predict house prices using location and property details")

# ---------------- HELPERS ----------------


def convert_sqft(x):
    try:
        x = str(x)
        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return np.nan

# ---------------- LOAD & TRAIN ----------------


@st.cache_resource
def load_model():
    df = pd.read_csv("data/Pune_House_Data.csv")

    # Keep only required columns
    df = df[["location", "total_sqft", "bhk", "bath", "balcony", "price"]]

    # Clean sqft
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df = df.dropna()

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"])

    X = df.drop("price", axis=1).astype(float)
    y = df["price"].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


model, columns = load_model()

# ---------------- UI ----------------
locations = [c.replace("location_", "")
             for c in columns if c.startswith("location_")]

location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet", 300, 10000, 600)
bhk = st.number_input("🏠 BHK", 1, 10, 2)
bath = st.number_input("🛁 Bathrooms", 1, 10, 2)
balcony = st.number_input("🌿 Balconies", 0, 10, 1)

# ---------------- PREDICT ----------------
if st.button("Predict Price"):
    x = np.zeros(len(columns))

    x[columns.get_loc("total_sqft")] = sqft
    x[columns.get_loc("bhk")] = bhk
    x[columns.get_loc("bath")] = bath
    x[columns.get_loc("balcony")] = balcony

    loc_col = f"location_{location}"
    if loc_col in columns:
        x[columns.get_loc(loc_col)] = 1

    price = model.predict([x])[0]
    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "End-to-End Machine Learning Project | Developed by Raam214"
    "</div>",
    unsafe_allow_html=True
)
