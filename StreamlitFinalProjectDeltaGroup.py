import streamlit as st
import pandas as pd
import numpy as np
import pickle
from custom_transformers import CustomImputer, FixPurchases, TotalMonetaryAdder

# Load pipeline
with open("clustering_pipeline.sav", "rb") as f:
    pipeline = pickle.load(f)

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Credit Card Customer Segmentation")

st.sidebar.title("⚙️ Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ["Single Entry", "Batch Upload"])

# Define cluster description (to be edited later)
cluster_info = pd.DataFrame({
    "Cluster": [0, 1, 2, 3],
    "Description": ["Mixed monetary with notable cash focused traits, moderate spenders, blended habits. Includes subscription payers and occasional shoppers. Low to moderate engagement users and not necessarily loyalty-driven or retail dependent. Less risky group with good payment behaviour.", 
                    "Understimulated retail dominant cluster but still behaviourally diverse (also use cash). Some users might be shifting from cash heavy usage to structured spending or vice versa. Less consistent repayment patterns. Occasional or seasonal engagement rather than loyalty-based spending. Possibly a mix of dormant users, seasonal spenders, and commited ones", 
                    "Extremely retail-focused group which means they are purchase-centric group. They are comfortable with structured borrowing and payback cycles. Loyal credit card shoppers. ", 
                    "Cash-centric user group. Low interaction with credit purchases and high preference for liquidity. Prioritise control, quick cash access, and avoid long-term financing. Which could be driven by income volatility/cashflow constraints, limited trust or interest in installment products, short term financial needs over long term planning. Low repayment habits with high utilization. Potential churn if not supported well -> Risky users."],
    "Recommendation": ["Cash perks,loyalty rewards, light engagement promos", 
                       "Depending on the purchase style and monetary style. If installments dominant and retail focused then merchant targetting. If oneoff dominant and mixed/retail focused then seasonal, gamified, or triggered promos. Lastly, if balanced style and cash focused, then cash perks", 
                       "Merchant partnerships, installment programs, or product bundling. Furthermore, for installment dominant users, use tiered loyalty offers based on credit engagement ; oneoff dominant users, use cashback campaigns and flash discounts; and balanced users, use flexible promos like A/B offers", 
                       "For high risk sub-segment, use fee relief, budgeting nudges, and repayment incentives. For low to moderate risk, use cash perks, loyalty rewards, and light engagement promos"]
})

html_table = "<style>td { max-width: 250px; word-wrap: break-word; white-space: normal; }</style>"
html_table += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
html_table += "<tr>" + "".join(f"<th>{col}</th>" for col in cluster_info.columns) + "</tr>"

for _, row in cluster_info.iterrows():
    html_table += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"

html_table += "</table>"

feature_order = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
]

def get_user_input():
    st.subheader("📋 Input Customer Information")
    input_data = {
        "BALANCE": st.number_input("Balance", min_value=0.0),
        "BALANCE_FREQUENCY": st.slider("Balance Frequency", 0.0, 1.0, 0.9),
        "PURCHASES": st.number_input("Total Purchases", min_value=0.0),
        "ONEOFF_PURCHASES": st.number_input("One-off Purchases", min_value=0.0),
        "INSTALLMENTS_PURCHASES": st.number_input("Installment Purchases", min_value=0.0),
        "CASH_ADVANCE": st.number_input("Cash Advance", min_value=0.0),
        "PURCHASES_FREQUENCY": st.slider("Purchases Frequency", 0.0, 1.0, 0.5),
        "ONEOFF_PURCHASES_FREQUENCY": st.slider("One-off Purchase Frequency", 0.0, 1.0, 0.3),
        "PURCHASES_INSTALLMENTS_FREQUENCY": st.slider("Installment Purchase Frequency", 0.0, 1.0, 0.4),
        "CASH_ADVANCE_FREQUENCY": st.slider("Cash Advance Frequency", 0.0, 1.0, 0.1),
        "CASH_ADVANCE_TRX": st.number_input("Cash Advance Transactions", min_value=0),
        "PURCHASES_TRX": st.number_input("Purchases Transactions", min_value=0),
        "CREDIT_LIMIT": st.number_input("Credit Limit", min_value=0.0),
        "PAYMENTS": st.number_input("Payments", min_value=0.0),
        "MINIMUM_PAYMENTS": st.number_input("Minimum Payments", min_value=0.0),
        "PRC_FULL_PAYMENT": st.slider("Percent Full Payment", 0.0, 1.0, 0.0),
        "TENURE": st.slider("Tenure (Months)", 0, 12, 12),
    }
    return pd.DataFrame([input_data])

def predict(df_input):
    predictions = pipeline.predict(df_input)
    return predictions

# --- Main interface ---
if input_mode == "Single Entry":
    df_single = get_user_input()
    if st.button("Predict Cluster"):
        cluster = predict(df_single)[0]
        st.success(f"Predicted Cluster: {cluster}")
        st.subheader("🧾 Cluster Summary")
        st.markdown(html_table, unsafe_allow_html=True)
else:
    st.subheader("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write("📊 Preview of Uploaded Data")
            st.dataframe(df_upload.head())

            if st.button("Predict Batch Clusters"):
                predictions = predict(df_upload)
                df_result = df_upload.copy()
                df_result["CLUSTER"] = predictions
                st.success("✅ Prediction complete!")
                st.subheader("🔢 Prediction Results")
                st.dataframe(df_result)

                st.subheader("🧾 Cluster Summary")
                st.markdown(html_table, unsafe_allow_html=True)

                # Option to download results
                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results CSV", csv, "cluster_results.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")