import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from preprocess import prep_data, file_data
from train import train_model, save_model
from evaluate import eval_model, load_model,model_explain

DATA_PATH  = "WineQT.csv"
MODEL_PATH = "model.pkl"

st.set_page_config(page_title="Wine Quality", page_icon="🍷")
st.title("🍷 Wine Quality Predictor")

# ── Preprocess ──
X_train, X_test, y_train, y_test, scaler = prep_data(DATA_PATH)

# ── Train & save if model doesn't exist ──
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time..."):
        model = train_model(X_train, X_test, y_train, y_test)
        save_model(model, scaler)
    st.success("Model trained and saved!")

model = load_model(MODEL_PATH)

tab1, tab2 = st.tabs(["🔮 Predict", "📈 Evaluate"])

# ── Tab 1: Predict ──
with tab1:
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Model Accuracy", f"{acc:.2%}")

    st.subheader("Enter Wine Properties")
    col1, col2, col3 = st.columns(3)
    fixed_acidity        = col1.number_input("Fixed Acidity",        value=7.4)
    volatile_acidity     = col2.number_input("Volatile Acidity",     value=0.70)
    citric_acid          = col3.number_input("Citric Acid",          value=0.00)
    residual_sugar       = col1.number_input("Residual Sugar",       value=1.9)
    chlorides            = col2.number_input("Chlorides",            value=0.065)
    free_sulfur_dioxide  = col3.number_input("Free Sulfur Dioxide",  value=15.0)
    total_sulfur_dioxide = col1.number_input("Total Sulfur Dioxide", value=21.0)
    density              = col2.number_input("Density",              value=0.9946, format="%.4f")
    pH                   = col3.number_input("pH",                   value=3.39)
    sulphates            = col1.number_input("Sulphates",            value=0.47)
    alcohol              = col2.number_input("Alcohol",              value=10.0)

    if st.button("Predict"):
        X = scaler.transform([[fixed_acidity, volatile_acidity, citric_acid,
                               residual_sugar, chlorides, free_sulfur_dioxide,
                               total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        if pred == 1:
            st.success(f"✅ High Quality Wine (≥7)  —  {proba:.1%} confidence")
        else:
            st.error(f"❌ Low Quality Wine (<7)  —  {1-proba:.1%} confidence")

with tab2:
    eval_model(model, X_test, y_test)
    st.pyplot(plt.gcf())
    plt.clf()
    model_explain(model, X_test)
    st.pyplot(plt.gcf())