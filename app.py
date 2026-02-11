import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Pune House Price Prediction",
    layout="centered"
)

st.title("🏠 Pune House Price Prediction")
st.caption("Predict house prices using location and property details")

# ===============================
# LOAD + TRAIN MODEL (CACHED)
# ===============================


@st.cache_resource
def load_model():
    df = pd.read_csv("data/Pune_House_Data.csv")

    X = df.drop("price", axis=1)
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


model, columns = load_model()

# ===============================
# USER INPUTS
# ===============================
location_cols = [c for c in columns if c.startswith("location_")]
locations = [c.replace("location_", "") for c in location_cols]

location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet", 300, 10000, 300)
bhk = st.number_input("🏠 BHK", 1, 10, 1)
bath = st.number_input("🛁 Bathrooms", 1, 10, 1)
balcony = st.number_input("🌿 Balconies", 0, 10, 0)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Price"):
    input_data = [0] * len(columns)

    input_data[columns.get_loc("total_sqft")] = sqft
    input_data[columns.get_loc("bhk")] = bhk
    input_data[columns.get_loc("bath")] = bath
    input_data[columns.get_loc("balcony")] = balcony

    loc_col = f"location_{location}"
    if loc_col in columns:
        input_data[columns.get_loc(loc_col)] = 1

    price = model.predict([input_data])[0]

    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")

st.markdown("---")
st.markdown("Powered by **Raam214 ❤️**")
