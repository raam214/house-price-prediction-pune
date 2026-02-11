import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pune House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Pune House Price Prediction")
st.caption("Predict house prices using location and property details")

# ---------------- LOAD & TRAIN MODEL ----------------


@st.cache_data
def load_data():
    df = pd.read_csv("data/Pune_House_Data.csv")
    df = df[["location", "total_sqft", "bath", "balcony", "price"]]
    df = df.dropna()

    X = pd.get_dummies(df.drop("price", axis=1))
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns, df


model, columns, df = load_data()

# ---------------- USER INPUT ----------------
location = st.selectbox("📍 Location", sorted(df["location"].unique()))
sqft = st.number_input("📐 Total Square Feet", min_value=300, value=1000)
bath = st.number_input("🛁 Bathrooms", min_value=1, value=2)
balcony = st.number_input("🌿 Balconies", min_value=0, value=1)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[location, sqft, bath, balcony]],
        columns=["location", "total_sqft", "bath", "balcony"]
    )

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn | Project by Raam214")
