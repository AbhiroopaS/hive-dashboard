import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import requests

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(page_title="Smart Hive System", layout="wide")

st.title("🐝 Smart Hive Monitoring & Treatment System")
st.markdown("### AI-Based Varroa Mite Detection + Weather-Based Treatment Decision")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in same folder

model = load_model()

# ----------------------------
# WEATHER FUNCTION
# ----------------------------

def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["main"]["temp"], data["main"]["humidity"]
    return None, None

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------

st.sidebar.header("🌍 Hive Environmental Conditions")

region = st.sidebar.text_input("Region", "Chennai")
api_key = st.sidebar.text_input("OpenWeather API Key", type="password")

if st.sidebar.button("🌦 Auto Fetch Weather"):
    if api_key:
        temp, hum = get_weather(region, api_key)
        if temp:
            st.sidebar.success("Weather fetched successfully ✅")
            temperature = temp
            humidity = hum
        else:
            st.sidebar.error("Weather fetch failed ❌")
            temperature = 30.0
            humidity = 60.0
    else:
        st.sidebar.warning("Enter API Key")
        temperature = 30.0
        humidity = 60.0
else:
    temperature = 30.0
    humidity = 60.0

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, float(temperature))
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, float(humidity))

# ----------------------------
# SESSION STORAGE
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# HIVE SETTINGS
# ----------------------------

num_hives = st.sidebar.number_input("Number of Hives", 1, 10, 2)
hive_ids = [f"Hive_{i+1}" for i in range(num_hives)]

# ----------------------------
# TREATMENT LOGIC
# ----------------------------

def recommend_treatment(total_mites, temp):

    if total_mites <= 2:
        return "LOW 🟢", "Preventive Care", "Sugar dusting or low-dose Thymol"

    elif 3 <= total_mites <= 5:
        if 15 <= temp <= 30:
            return "MEDIUM 🟡", "Thymol (12–15 g)", "Repeat after 14 days"
        else:
            return "MEDIUM 🟡", "Thymol (Temp Warning)", "Best between 15–30°C"

    else:
        if 10 <= temp <= 25:
            return "HIGH 🔴", "Formic Acid (60–65%)", "40–50 mL for 7–14 days"
        else:
            return "HIGH 🔴", "Formic Acid (Temp Warning)", "Optimal 10–25°C"

# ----------------------------
# INFECTION DISPLAY
# ----------------------------

def infection_display(mites):
    st.write("### 🧪 Infection Severity (Scale 0–10)")
    progress = min(mites / 10, 1.0)

    if mites <= 2:
        st.success(f"LOW Infection ({mites}/10)")
    elif 3 <= mites <= 5:
        st.warning(f"MEDIUM Infection ({mites}/10)")
    else:
        st.error(f"HIGH Infection ({mites}/10)")

    st.progress(progress)

# ============================
# HIVE PROCESSING
# ============================

for hive in hive_ids:

    st.markdown("---")
    st.subheader(f"🏠 {hive}")

    capture_count = st.number_input(
        f"Number of Bees (default 5)",
        min_value=1,
        max_value=20,
        value=5,
        key=f"{hive}_count"
    )

    mode = st.radio(
        "Select Image Source",
        ["📂 Upload Images", "📷 Use Camera"],
        key=f"{hive}_mode"
    )

    total_mites = 0
    total_images = 0

    # ------------------------
    # FILE UPLOAD
    # ------------------------

    if mode == "📂 Upload Images":

        uploaded_files = st.file_uploader(
            f"Upload images for {hive}",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"{hive}_upload"
        )

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
                total_images += 1
                cols[idx % 4].image(annotated, width=180)

    # ------------------------
    # CAMERA MODE
    # ------------------------

    if mode == "📷 Use Camera":

        st.info(f"Capture {capture_count} bee images")

        for i in range(capture_count):

            camera_image = st.camera_input(
                f"Capture Bee {i+1}",
                key=f"{hive}_camera_{i}"
            )

            if camera_image:

                image = cv2.imdecode(
                    np.frombuffer(camera_image.getvalue(), np.uint8),
                    cv2.IMREAD_COLOR
                )

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
                total_images += 1

                st.image(annotated, caption=f"Mites: {mite_count}", width=300)

                if st.button(f"🔄 Retake Bee {i+1}", key=f"{hive}_retake_{i}"):
                    st.rerun()

    # ------------------------
    # SHOW RESULTS
    # ------------------------

    if total_images > 0:

        infection, treatment, note = recommend_treatment(total_mites, temperature)

        st.write(f"🦠 Total Mites (per {total_images} bees): {total_mites}")
        infection_display(total_mites)

        st.write(f"⚠ Infection Level: {infection}")
        st.write(f"🧪 Recommended Treatment: {treatment}")
        st.write(f"📌 Note: {note}")

        st.session_state.history.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hive ID": hive,
            "Total Mites": total_mites,
            "Infection": infection,
            "Treatment": treatment,
            "Temperature": temperature,
            "Humidity": humidity
        })

# ============================
# HISTORY SECTION
# ============================

st.markdown("---")
st.header("📊 Hive History & Analytics")

history_df = pd.DataFrame(st.session_state.history)

if not history_df.empty:

    st.line_chart(history_df.set_index("Date")["Total Mites"])

    st.dataframe(history_df, use_container_width=True)

    output = BytesIO()
    history_df.to_excel(output, index=False)

    st.download_button(
        label="📥 Download Hive Report",
        data=output.getvalue(),
        file_name="Hive_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
