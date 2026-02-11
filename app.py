import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pune House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open("model/pune_house_price_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = list(data["columns"])  # IMPORTANT: convert Index → list

# ---------------- TITLE ----------------
st.title("🏠 Pune House Price Prediction")
st.write("Predict house prices using location and property details")

st.markdown("---")

# ---------------- INPUTS ----------------
location = st.selectbox(
    "📍 Location",
    sorted([col.replace("location_", "")
           for col in columns if col.startswith("location_")])
)

sqft = st.number_input("📐 Total Square Feet",
                       min_value=200, max_value=10000, value=300)
bhk = st.number_input("🏠 BHK", min_value=1, max_value=10, value=1)
bath = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=1)
balcony = st.number_input("🌿 Balconies", min_value=0, max_value=5, value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    input_data = np.zeros(len(columns))

    # numeric values
    input_data[columns.index("total_sqft")] = sqft
    input_data[columns.index("bath")] = bath
    input_data[columns.index("balcony")] = balcony
    input_data[columns.index("bhk")] = bhk

    # location one-hot
    loc_col = f"location_{location}"
    if loc_col in columns:
        input_data[columns.index(loc_col)] = 1

    prediction = model.predict([input_data])[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Powered by **Raam214 ❤️**")
