import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

DATA_FILE = Path('E:\Waddah\Alex\VS\Metho\DS Methodology Final Project\Dataset\data_without_outliers.csv').with_name("data_without_outliers.csv")
RF_MODEL_FILE = Path('E:\Waddah\Alex\VS\Metho\DS Methodology Final Project\Models\Random_Forest_Model.joblib').with_name("Random_Forest_Model.joblib")
XGB_MODEL_FILE = Path('E:\Waddah\Alex\VS\Metho\DS Methodology Final Project\Models\XGBoost_Model.joblib').with_name("XGBoost_Model.joblib")


@st.cache_data

def load_feature_metadata():
    df = pd.read_csv(DATA_FILE)
    obj_cols = df.select_dtypes(include=["object"]).columns
    data = df.drop(columns=obj_cols)
    target_col = "Price"
    feature_cols = [c for c in data.columns if c != target_col]
    stats = data[feature_cols].describe().T  # for sensible defaults

    return feature_cols, stats

@st.cache_resource

def load_models():
    rf_model = joblib.load(RF_MODEL_FILE)
    xgb_model = joblib.load(XGB_MODEL_FILE)
    return rf_model, xgb_model

def main():
    st.set_page_config(
        page_title="Laptop Price Prediction",
        layout="centered",
    )

    st.title("Laptop Price Prediction")

    try:
        feature_cols, stats = load_feature_metadata()

    except FileNotFoundError:
        st.error(
            "Could not find `data_without_outliers.csv`. "
            "Make sure this file is in the same folder as this app."
        )
        return
    try:
        rf_model, xgb_model = load_models()

    except FileNotFoundError:
        st.error(
            "Couldnt find one or both model files "
        )
        return

    st.subheader("Input Features")

    with st.form("prediction_form"):
        user_values = {}
        for feature in feature_cols:
            col_stats = stats.loc[feature]
            default_val = float(col_stats["mean"])
            user_values[feature] = st.number_input(
                feature,
                value=default_val,
                help=f"Typical range: {col_stats['min']:.2f} – {col_stats['max']:.2f}",
            )

        submitted = st.form_submit_button("Predict Price")
    if submitted:
        X_new = pd.DataFrame([user_values])[feature_cols]
        rf_pred = float(rf_model.predict(X_new)[0]) - 10000
        xgb_pred = float(xgb_model.predict(X_new)[0]) - 10000
        avg_pred = (rf_pred + xgb_pred) / 2.0
        st.subheader("Predicted Price")
        col1, col2, col3 = st.columns(3)
        col1.metric("Random Forest", f"{rf_pred:,.0f}")
        col2.metric("XGBoost", f"{xgb_pred:,.0f}")
        col3.metric("Average", f"{avg_pred:,.0f}")
        st.caption(
            "Made by *Waddah, Hammam, Eman, and Mnahel*"
        )

if __name__ == "__main__":

    main()





