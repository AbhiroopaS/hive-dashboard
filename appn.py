import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import random

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(page_title="Smart Hive System", layout="wide")

st.image("hive_banner.png", use_container_width=True)
st.title("🐝 Smart Hive Monitoring & Treatment System")
st.markdown("### AI-Based Varroa Mite Detection & Smart Treatment Recommendation")

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------

st.sidebar.header("🌍 Overall Hive Conditions")

region = st.sidebar.text_input("Region", "Chennai")
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 30.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
brood_presence = st.sidebar.selectbox("Brood Presence", ["Yes", "No"])

if st.sidebar.button("Simulate Live Sensor Data"):
    temperature = random.randint(25, 40)
    humidity = random.randint(40, 80)

# ----------------------------
# SESSION STORAGE
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

hive_ids = ["I1", "I2"]

# ----------------------------
# TREATMENT LOGIC
# ----------------------------

def recommend_treatment(total_mites, total_images, temp):

    if total_images == 0:
        return "NO DATA", "No Treatment", "Upload images first."

    mites_per_10 = total_mites

    if mites_per_10 <= 2:
        infection = "LOW 🟢"
        treatment = "Preventive Care"
        note = "Use sugar dusting or low-dose Thymol as precaution."

    elif 3 <= mites_per_10 <= 5:
        infection = "MEDIUM 🟡"
        if 15 <= temp <= 30:
            treatment = "Thymol (12–15 g)"
            note = "Repeat after 14 days."
        else:
            treatment = "Thymol (Temp Warning)"
            note = "Best applied between 15–30°C."

    else:
        infection = "HIGH 🔴"
        if 10 <= temp <= 25:
            treatment = "Formic Acid (60–65%)"
            note = "40–50 mL for 7–14 days."
        else:
            treatment = "Formic Acid (Temp Warning)"
            note = "Optimal range 10–25°C."

    return infection, treatment, note

# ----------------------------
# SIMPLE INFECTION DISPLAY
# ----------------------------

def infection_display(mites):

    st.write("### 🧪 Infection Severity (Scale 0–10)")

    progress = min(mites / 10, 1.0)

    if mites <= 2:
        st.success(f"🟢 LOW Infection ({mites}/10)")
    elif 3 <= mites <= 5:
        st.warning(f"🟡 MEDIUM Infection ({mites}/10)")
    else:
        st.error(f"🔴 HIGH Infection ({mites}/10)")

    st.progress(progress)

# ----------------------------
# HIVE PROCESSING
# ----------------------------

for hive in hive_ids:

    st.markdown("---")
    st.subheader(f"🏠 Hive {hive}")

    uploaded_files = st.file_uploader(
        f"Upload images for Hive {hive}",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key=hive
    )

    total_mites = 0

    if uploaded_files:

        cols = st.columns(4)

        for idx, uploaded_file in enumerate(uploaded_files):

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = model(image)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            mite_count = 0
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name.lower() == "mite":
                    mite_count += 1

            total_mites += mite_count
            cols[idx % 4].image(annotated, width=180)

        infection, treatment, note = recommend_treatment(
            total_mites,
            len(uploaded_files),
            temperature
        )

        st.write(f"🦠 Total Mites (per {len(uploaded_files)} bees): {total_mites}")

        infection_display(total_mites)

        st.write(f"⚠ Infection Level: {infection}")
        st.write(f"🧪 Recommended Treatment: {treatment}")
        st.write(f"📌 Note: {note}")

        if infection == "LOW 🟢":
            st.warning("⚠ Low infection – Preventive treatment advised.")
        elif infection == "MEDIUM 🟡":
            st.warning("⚠ Moderate infection – Treatment required.")
        else:
            st.error("🚨 HIGH INFECTION – Immediate treatment required!")

        st.session_state.history.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hive ID": hive,
            "Mites (per 10 bees)": total_mites,
            "Infection": infection,
            "Treatment": treatment
        })

# ----------------------------
# HISTORY SECTION
# ----------------------------

st.markdown("---")
st.header("📊 Hive History & Analytics")

history_df = pd.DataFrame(st.session_state.history)

if not history_df.empty:

    selected_hive = st.selectbox("Select Hive for Trend", hive_ids)

    hive_trend = history_df[history_df["Hive ID"] == selected_hive]

    if not hive_trend.empty:
        st.line_chart(hive_trend.set_index("Date")["Mites (per 10 bees)"])

    st.subheader("🧪 Treatment History Log")
    st.dataframe(history_df, use_container_width=True)

    output = BytesIO()
    history_df.to_excel(output, index=False)

    st.download_button(
        label="📥 Download Full Historical Report",
        data=output.getvalue(),
        file_name="Hive_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )