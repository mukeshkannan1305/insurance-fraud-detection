import streamlit as st
import pandas as pd
import numpy as np
import cv2
import re
from PIL import Image
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import easyocr
from sklearn.cluster import KMeans

# ----------------------------
# LOAD OCR
# ----------------------------
reader = easyocr.Reader(['en'])

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="Multimodel Insurance Fraud Detection Using AI", layout="wide")

# ----------------------------
# SESSION STATE
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "claim_submitted" not in st.session_state:
    st.session_state.claim_submitted = False

# ----------------------------
# LOAD USERS
# ----------------------------
@st.cache_data
def load_users():
    return pd.read_csv("user.csv")

users = load_users()

# ----------------------------
# TRAIN MODEL
# ----------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("dataset.csv")

    X = data.drop(["fraud", "customer_id"], axis=1)
    y = data["fraud"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, accuracy, feature_names

model, scaler, accuracy, feature_names = train_model()

# ----------------------------
# DAMAGE DETECTION
# ----------------------------
def analyze_damage(image):
    img = np.array(image)
    img = cv2.resize(img, (640, 480))

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    damage_ratio = np.sum(edges > 0) / (img.shape[0] * img.shape[1])

    if damage_ratio > 0.10:
        return 5
    elif damage_ratio > 0.07:
        return 4
    elif damage_ratio > 0.04:
        return 3
    elif damage_ratio > 0.02:
        return 2
    else:
        return 1

# ----------------------------
# FIX OCR TEXT
# ----------------------------
def fix_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    return text

# ----------------------------
# 🔥 FINAL NUMBER PLATE DETECTION
# ----------------------------
def extract_number_plate(image):
    img = np.array(image)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (640, 480))
    h, w, _ = img.shape

    # MULTIPLE CROPS
    crops = [
        img[int(h*0.4):int(h*0.75), int(w*0.3):int(w*0.95)],
        img[int(h*0.45):int(h*0.8), int(w*0.2):int(w*0.9)],
        img[int(h*0.35):int(h*0.7), int(w*0.4):int(w*1.0)]
    ]

    pattern = r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}"

    for i, crop in enumerate(crops):

        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)

        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # DEBUG VIEW
        st.image(thresh, caption=f"🔍 OCR Crop {i+1}")

        # PASS 1: THRESHOLD
        results = reader.readtext(thresh)

        for (_, text, prob) in results:
            if prob < 0.3:
                continue

            cleaned = fix_plate(text)

            if re.search(pattern, cleaned):
                return cleaned

        # PASS 2: ORIGINAL
        results = reader.readtext(crop)

        for (_, text, prob) in results:
            if prob < 0.3:
                continue

            cleaned = fix_plate(text)

            if re.search(pattern, cleaned):
                return cleaned

    return None

# ----------------------------
# COLOR DETECTION
# ----------------------------
def detect_car_color(image):
    img = np.array(image)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    h, w, _ = img.shape
    crop = img[int(h*0.55):int(h*0.85), int(w*0.4):int(w*0.95)]

    pixels = crop.reshape(-1, 3)

    kmeans = KMeans(n_clusters=2, random_state=42).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    r, g, b = dominant

    if r < 80 and g < 80 and b < 80:
        return "black"
    elif r > 150 and g < 100:
        return "red"
    elif b > 120:
        return "blue"
    else:
        return "grey"

# ----------------------------
# TEXT RISK
# ----------------------------
def text_risk(text):
    words = ["maybe", "not sure", "approx", "guess"]
    return sum(w in text.lower() for w in words) * 0.2

# ----------------------------
# LOGIN
# ----------------------------
if not st.session_state.logged_in:

    st.title("🔐 Login")

    uid = st.text_input("User ID")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        user = users[
            (users["user_id"].astype(str) == uid) &
            (users["password"].astype(str) == pwd)
        ]

        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.user = user.iloc[0]
            st.success("Login Success")
            st.rerun()
        else:
            st.error("Invalid login")

# ----------------------------
# MAIN APP
# ----------------------------
else:
    user = st.session_state.user

    st.title("🚗 Multimodel insurance Detection Using AI")

    st.write("Welcome:", user["name"])
    st.write("Registered Plate:", user["plate"])
    st.write("Registered Color:", user["color"])

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if not st.session_state.claim_submitted:

        st.header("Apply Claim")

        age = st.slider("Age", 18, 80, 30)
        years = st.slider("Policy Years", 1, 10, 3)
        text = st.text_area("Explain Damage")
        file = st.file_uploader("Upload Image")

        if st.button("Submit"):
            if file:
                st.session_state.image = Image.open(file)
                st.session_state.age = age
                st.session_state.years = years
                st.session_state.text = text
                st.session_state.claim_submitted = True
                st.rerun()

    else:
        img = st.session_state.image
        st.image(img)

        with st.spinner("Processing..."):
            damage = analyze_damage(img)
            plate = extract_number_plate(img)
            color = detect_car_color(img)

        st.metric("Detected Plate", plate if plate else "Not Found")
        st.metric("Detected Color", color)
        st.metric("Damage Level", damage)

        if plate is None:
            st.error("❌ Plate not detected")
            st.stop()

        if plate != user["plate"]:
            st.error("❌ Plate mismatch")
            st.stop()

        if color != user["color"]:
            st.error("❌ Color mismatch")
            st.stop()

        st.success(f"✔ Plate Verified: {plate}")

        # ML
        cost = 50000 * damage

        data = pd.DataFrame([{
            "age": st.session_state.age,
            "claim_amount": cost,
            "policy_years": st.session_state.years,
            "past_claims": int(user.get("past_claims", 0)),
            "accident_history": int(user.get("accident_history", 0)),
            "damage_severity": damage,
            "text_risk": text_risk(st.session_state.text),
            "image_risk": damage / 5
        }])

        for col in feature_names:
            if col not in data:
                data[col] = 0

        data = data[feature_names]

        scaled = scaler.transform(data)
        prob = model.predict_proba(scaled)[0][1]

        st.progress(float(prob))
        st.write(f"Fraud Probability: {prob*100:.2f}%")

        if prob >= 0.6:
            st.error("❌ CLAIM REJECTED")
        else:
            st.success("✅ CLAIM APPROVED")
            st.write(f"Amount: ₹{cost:,}")

        st.write("Model Accuracy:", accuracy)

        if st.button("New Claim"):
            st.session_state.claim_submitted = False
            st.rerun()
