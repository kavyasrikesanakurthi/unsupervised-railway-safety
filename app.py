import streamlit as st
import pandas as pd
import joblib
import sqlite3
import hashlib
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Railway Safety Risk System",
    page_icon="🚉",
    layout="wide"
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# ---------------- PASSWORD HASH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- AUTH FUNCTIONS ----------------
def login_user(username, password):
    hashed = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hashed))
    return c.fetchone()

def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users(username, password) VALUES (?,?)",
                  (username, hashed))
        conn.commit()
        return True
    except:
        return False

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.title("🚉 Railway Station Safety Risk System")

    option = st.selectbox("Select Option", ["Login", "Signup"])

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid Credentials")

    else:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Signup"):
            if signup_user(new_user, new_pass):
                st.success("Account Created Successfully! Please Login.")
            else:
                st.error("Username already exists")

# ---------------- DASHBOARD ----------------
else:

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.title("🚉 Railway Safety Risk Dashboard")
    st.markdown("Upload dataset to analyze station safety risk patterns.")

    uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"])

    if uploaded_file is not None:

        try:
            # Load trained objects
            scaler = joblib.load("scaler.pkl")
            pca = joblib.load("pca.pkl")
            kmeans = joblib.load("kmeans_model.pkl")

            raw = pd.read_csv(uploaded_file)

            # Keep only numeric columns
            numeric_data = raw.select_dtypes(include=["number"])

            # Check feature match
            if numeric_data.shape[1] != scaler.n_features_in_:
                st.error("Uploaded dataset does not match training features.")
                st.stop()

            # Scale
            scaled = scaler.transform(numeric_data)

            # PCA transform
            pca_data = pca.transform(scaled)

            # Predict cluster
            clusters = kmeans.predict(pca_data)

            # Distance from centroid
            distances = kmeans.transform(pca_data)
            risk_score = distances.min(axis=1)

            raw["Risk_Score"] = risk_score

            # -------- STRICT 2 LEVEL CLASSIFICATION --------
            threshold = np.median(risk_score)

            raw["Risk_Level"] = np.where(
                raw["Risk_Score"] > threshold,
                "High Risk",
                "Low Risk"
            )

            # Force remove any accidental values
            raw["Risk_Level"] = raw["Risk_Level"].replace(
                ["Medium Risk"], "Low Risk"
            )

            # Station ID
            raw["Station_ID"] = [
                "STN_" + str(i+1).zfill(4) for i in range(len(raw))
            ]

            # ---------------- METRICS ----------------
            total = len(raw)
            high = (raw["Risk_Level"] == "High Risk").sum()
            low = (raw["Risk_Level"] == "Low Risk").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Stations", total)
            col2.metric("High Risk Stations", high)
            col3.metric("Low Risk Stations", low)

            st.divider()

            # ---------------- BAR GRAPH ----------------
            risk_counts = raw["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]

            # Ensure only 2 categories shown
            risk_counts = risk_counts[risk_counts["Risk_Level"].isin(["High Risk", "Low Risk"])]

            fig = px.bar(
                risk_counts,
                x="Risk_Level",
                y="Count",
                text="Count",
                color="Risk_Level",
                category_orders={"Risk_Level": ["High Risk", "Low Risk"]}
            )

            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ---------------- TOP HIGH RISK ----------------
            st.subheader("🔴 Top 20 High Risk Stations")

            top_risk = raw[raw["Risk_Level"] == "High Risk"] \
                .sort_values("Risk_Score", ascending=False) \
                .head(20)

            st.dataframe(
                top_risk[["Station_ID", "Risk_Level", "Risk_Score"]],
                use_container_width=True
            )

            st.divider()

            # ---------------- INTERPRETATION ----------------
            st.subheader("📌 System Interpretation")

            st.info(
                "Risk is determined by distance from cluster centroid. "
                "Stations far from normal operational pattern are classified as High Risk."
            )

            st.error("🔴 High Risk → Immediate inspection recommended.")
            st.success("🟢 Low Risk → Routine monitoring sufficient.")

            # ---------------- DOWNLOAD ----------------
            report = raw[["Station_ID", "Risk_Level", "Risk_Score"]]

            st.download_button(
                "Download Risk Report",
                report.to_csv(index=False),
                "Railway_Risk_Report.csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Please upload dataset to begin analysis.")