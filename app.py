import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Pune House Price Prediction",
    layout="centered"
)

st.title("🏠 Pune House Price Prediction")
st.caption("Predict house prices using location and property details")

# ----------------------------------
# LOAD DATA & TRAIN MODEL
# ----------------------------------


@st.cache_resource
def load_model():
    df = pd.read_csv("data/Pune_House_Data.csv")

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


model, columns = load_model()

# ----------------------------------
# INPUT UI
# ----------------------------------
location_columns = [c for c in columns if c.startswith("location_")]
locations = [c.replace("location_", "") for c in location_columns]

location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet",
                       min_value=300, max_value=10000, value=300)
bhk = st.number_input("🏠 BHK", min_value=1, max_value=10, value=1)
bath = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, value=1)
balcony = st.number_input("🌿 Balconies", min_value=0, max_value=10, value=0)

# ----------------------------------
# PREDICTION
# ----------------------------------
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

# ----------------------------------
# FOOTER (PROFESSIONAL)
# ----------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Developed by <b>Raam214</b> | Machine Learning Project"
    "</div>",
    unsafe_allow_html=True
)
