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
st.image("hive_banner.png", use_container_width=True)
st.title("🐝 Smart Hive Monitoring & Treatment System")
st.markdown("### AI-Based Varroa Mite Detection + Weather-Based Treatment Decision")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------
# WEATHER FUNCTION
# ----------------------------

def get_weather(city, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["main"]["temp"], data["main"]["humidity"]
    except:
        pass
    return None, None

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------

st.sidebar.header("🌍 Hive Environmental Conditions")

region = st.sidebar.text_input("Region", "Chennai")
api_key = "50d09d1e6b4bcec3015df459ecbfa113"

if st.sidebar.button("🌦 Auto Fetch Weather"):
    temp, hum = get_weather(region, api_key)
    if temp is not None:
        temperature = temp
        humidity = hum
        st.sidebar.success("Weather fetched successfully ✅")
    else:
        temperature = 30.0
        humidity = 60.0
        st.sidebar.error("Weather fetch failed ❌")
else:
    temperature = 30.0
    humidity = 60.0

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, float(temperature))
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, float(humidity))

season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
brood_status = st.sidebar.radio("Brood Status", ["Present", "Absent"])

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

def recommend_treatment(total_mites, total_bees, temp, season, brood_status):

    infection_percent = (total_mites / total_bees) * 10

    # LOW
    if infection_percent < 3:
        return "LOW 🟢", infection_percent, {
            "Treatment": "Preventive Monitoring",
            "Dosage": "No chemical required",
            "Duration": "Re-check after 2 weeks",
            "Brood Safe": "Yes",
            "Warning": "Maintain hive hygiene"
        }

    # MEDIUM
    elif 3 <= infection_percent <= 7:

        if 15 <= temp <= 30 and season in ["Spring", "Autumn"]:
            return "MEDIUM 🟡", infection_percent, {
                "Treatment": "Thymol (12–15 g)",
                "Dosage": "12–15 g",
                "Duration": "Repeat every 14 days (4–6 weeks)",
                "Brood Safe": "Partial",
                "Warning": "Avoid >30°C"
            }

        elif 10 <= temp <= 25 and season in ["Summer", "Autumn"]:
            return "MEDIUM 🟡", infection_percent, {
                "Treatment": "Formic Acid (60–65%)",
                "Dosage": "20–30 mL",
                "Duration": "7–14 days",
                "Brood Safe": "Yes",
                "Warning": "Monitor queen"
            }

        else:
            return "MEDIUM 🟡", infection_percent, {
                "Treatment": "Environmental Conditions Not Ideal",
                "Dosage": "Wait",
                "Duration": "Monitor",
                "Brood Safe": "-",
                "Warning": "Adjust temperature or season"
            }

    # HIGH
    else:

        if brood_status == "Present":

            if 10 <= temp <= 25:
                return "HIGH 🔴", infection_percent, {
                    "Treatment": "Formic Acid (60–65%)",
                    "Dosage": "40–50 mL",
                    "Duration": "7–14 days",
                    "Brood Safe": "Yes (works in capped brood)",
                    "Warning": "Monitor queen carefully"
                }
            else:
                return "HIGH 🔴", infection_percent, {
                    "Treatment": "Formic Acid Recommended",
                    "Dosage": "40–50 mL",
                    "Duration": "7–14 days",
                    "Brood Safe": "Yes",
                    "Warning": "Temperature not ideal (10–25°C)"
                }

        else:

            if temp < 15:
                return "HIGH 🔴", infection_percent, {
                    "Treatment": "Oxalic Acid (3.2–3.5%)",
                    "Dosage": "5 mL per seam",
                    "Duration": "Single application",
                    "Brood Safe": "Broodless only",
                    "Warning": "Apply once"
                }
            else:
                return "HIGH 🔴", infection_percent, {
                    "Treatment": "Oxalic Acid Recommended",
                    "Dosage": "5 mL per seam",
                    "Duration": "Single application",
                    "Brood Safe": "Broodless only",
                    "Warning": "Best below 15°C"
                }

# ----------------------------
# HIVE PROCESSING
# ----------------------------

for hive in hive_ids:

    st.markdown("---")
    st.subheader(f"🏠 {hive}")

    capture_count = st.number_input(
        "Number of Bees (default 5)",
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

    # Upload Mode
    if mode == "📂 Upload Images":

        uploaded_files = st.file_uploader(
            f"Upload images for {hive}",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"{hive}_upload"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:

                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                results = model(image)

                mite_count = 0
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        if class_name.lower() == "mite":
                            mite_count += 1

                total_mites += mite_count
                total_images += 1

    # Camera Mode
    if mode == "📷 Use Camera":

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

                mite_count = 0
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        if class_name.lower() == "mite":
                            mite_count += 1

                total_mites += mite_count
                total_images += 1

    if total_images > 0:

        level, percent, treatment_data = recommend_treatment(
            total_mites,
            total_images,
            temperature,
            season,
            brood_status
        )

        st.write(f"🦠 Total Mites: {total_mites}")
        st.write(f"📊 Infection Percentage: {percent:.2f}%")
        st.write(f"⚠ Infection Level: {level}")

        st.subheader("🧪 Treatment Recommendation")
        st.write(f"**Treatment:** {treatment_data['Treatment']}")
        st.write(f"**Dosage:** {treatment_data['Dosage']}")
        st.write(f"**Duration:** {treatment_data['Duration']}")
        st.write(f"**Brood Compatibility:** {treatment_data['Brood Safe']}")
        st.write(f"**Warning:** {treatment_data['Warning']}")

        st.session_state.history.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hive ID": hive,
            "Total Mites": total_mites,
            "Infection %": round(percent, 2),
            "Infection Level": level,
            "Treatment": treatment_data["Treatment"],
            "Temperature": temperature,
            "Humidity": humidity,
            "Season": season,
            "Brood Status": brood_status
        })

# ----------------------------
# HISTORY SECTION
# ----------------------------

st.markdown("---")
st.header("📊 Hive History & Analytics")

history_df = pd.DataFrame(st.session_state.history)

if not history_df.empty:

    history_df["Date"] = pd.to_datetime(history_df["Date"])
    history_df = history_df.sort_values("Date")

    st.line_chart(history_df.set_index("Date")["Infection %"])
    st.dataframe(history_df, use_container_width=True)

    output = BytesIO()
    history_df.to_excel(output, index=False)

    st.download_button(
        label="📥 Download Hive Report",
        data=output.getvalue(),
        file_name="Hive_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hive data recorded yet.")
