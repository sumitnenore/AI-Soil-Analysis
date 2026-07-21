import json
import re
import numpy as np
import streamlit as st
from PIL import Image
from AskAi.Askai import askai
import os

try:
    from tensorflow.keras import models 
except Exception:
    models = None

model = None
model_loaded = False

if models is not None:
    try:
        MODEL_PATH = os.path.join("Model", "SoilModel.h5")
        model = models.load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        st.warning(f"Could not load the model file: {e}")
else:
    st.warning("TensorFlow is not installed in this environment, so prediction is unavailable for now.")

st.set_page_config(page_title="SoilWise AI", page_icon="🌿", layout="wide")

with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <h1>🌿 AI Soil & Crop Advisor</h1>
            <p>Upload a picture of your land's soil, input your farm details, and our deep learning model combined with GenAI will recommend the best crops for you.</p>
            <div class="hero-pills">
                <span class="pill">AI Soil Prediction</span>
                <span class="pill">Smart Crop Matching</span>
                <span class="pill">Farming Insights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='metric-box'><strong>🖼️ Visual Soil Analysis</strong><span>Identify soil type from field images with high confidence and a polished dashboard experience.</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='metric-box'><strong>🌾 Crop Planning</strong><span>Match crops to soil, weather, rainfall, land size, and budget for better decisions.</span></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='metric-box'><strong>💡 Actionable Advice</strong><span>Receive practical recommendations for farming practices, profitability, and risk management.</span></div>",
            unsafe_allow_html=True,
        )


def parse_ai_response(raw_output):
    if not raw_output:
        return None
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def ensure_list(value):
    """Convert farming_practices/risks to a proper list of complete items."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        items = re.split(r"\.\s+(?=[A-Z])|;\s*|,\s*|\n+", text)
        return [item.strip() for item in items if item.strip()]
    return [str(value)] if value else []


def render_results(result, predicted_soil):
    if not result:
        st.markdown("<div class='empty-state'>The AI report is empty. Please try again.</div>", unsafe_allow_html=True)
        return

    soil = result.get("soil_analysis", {}) or {}
    crops = result.get("recommended_crops", []) or []
    if not isinstance(crops, list):
        crops = [crops]
    practices = ensure_list(result.get("farming_practices", []))
    risks = ensure_list(result.get("possible_risks", []))
    seed_qty = result.get("seed_quantity_required", {}) or {}
    most_profitable = result.get("most_profitable_crop", "N/A")

    # AI Diagnosis & Agricultural Report Header
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e7f45 0%, #49b96d 100%); color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0;">
            <h2 style="margin: 0; font-size: 1.8rem;">📊 AI Diagnosis & Agricultural Report</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Soil Prediction Info
    st.markdown(
        f"""
        <div class='result-card'>
            <h3>🌍 Soil Type Used for Analysis</h3>
            <div style="background: linear-gradient(135deg, #1e7f45 0%, #49b96d 100%); color: white; padding: 0.8rem; border-radius: 12px; margin-top: 0.5rem; margin-bottom: 0.5rem;">
                <strong style="font-size: 1.2rem;">{predicted_soil}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Soil Analysis Details
    st.markdown(
        f"""
        <div class='result-card' style="margin-top: 1rem;">
            <h3>🔍 Soil Analysis Details</h3>
            <p><strong>Condition:</strong> {soil.get('soil_condition', '-')}</p>
            <p><strong>Suitability:</strong> {soil.get('suitability', '-')}</p>
            <p><strong>Weather Suitability:</strong> {soil.get('weather_suitability', '-')}</p>
            <div class='metric-row'>
                <div class='mini-stat'><span>Moisture Level</span><strong>{soil.get('moisture_level', '-')}</strong></div>
                <div class='mini-stat'><span>Fertility Rating</span><strong>{soil.get('fertility_rating', '-')}/10</strong></div>
            </div>
            <div class='metric-row'>
                <div class='mini-stat'><span>Texture</span><strong>{soil.get('key_observations', {}).get('texture', '-')}</strong></div>
                <div class='mini-stat'><span>Color</span><strong>{soil.get('key_observations', {}).get('color', '-')}</strong></div>
                <div class='mini-stat'><span>Compactness</span><strong>{soil.get('key_observations', {}).get('compactness', '-')}</strong></div>
                <div class='mini-stat'><span>Drainage</span><strong>{soil.get('key_observations', {}).get('drainage', '-')}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Matched Recommended Crop
    if crops:
        top_crop = crops[0]
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1e7f45 0%, #49b96d 100%); color: white; padding: 1.2rem; border-radius: 20px; margin: 1.5rem 0;">
                <h3 style="margin: 0 0 0.3rem 0; font-size: 1.1rem;">⭐ TOP-MATCHED RECOMMENDED CROP</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2.2rem;">🌾 {top_crop.get('crop_name', 'Crop')}</h2>
                <p style="margin: 0.5rem 0; font-size: 1rem; line-height: 1.6;">{top_crop.get('reason', '-')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top 5 Cultivation Recommendations
    st.markdown(
        """
        <div style="margin: 1.5rem 0;">
            <h2 style="color: #163426; margin-bottom: 1rem;">🌱 Top 5 Cultivation Recommendations</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    cols = st.columns(5)
    for idx, crop in enumerate(crops[:5]):
        with cols[idx]:
            st.markdown(
                f"""
                <div class='crop-card' style="padding: 1rem; text-align: center; background: linear-gradient(135deg, #12202a 0%, #1d303d 100%); border: 1px solid rgba(77, 185, 128, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌾</div>
                    <h4 style="margin: 0.3rem 0; color: #f4f7fb;">{crop.get('crop_name', f'Crop {idx+1}')}</h4>
                    <p style="margin: 0.3rem 0; font-size: 0.85rem; color: rgba(244, 247, 251, 0.75);">{crop.get('best_months_to_grow', '-')}</p>
                    <div class='score-pill' style="justify-content: center; margin: 0.5rem 0;">⭐ Score: {crop.get('suitability_score', '-')}/5.0</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Detailed Crop Cards
    st.markdown(
        """
        <div style="margin: 1.5rem 0;">
            <h2 style="color: #163426; margin-bottom: 1rem;">📋 Detailed Crop Information</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, crop in enumerate(crops):
        with st.expander(f"🌾 {crop.get('crop_name', f'Crop {idx+1}')} - Score: {crop.get('suitability_score', '-')}/10", expanded=(idx==0)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="background: #162630; color: #edf2f7; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(77, 185, 128, 0.16);">
                        <h4 style="margin-top: 0; color: #f4f7fb;">📌 Basic Information</h4>
                        <p style="margin: 0.2rem 0;"><strong>Crop Name:</strong> {crop.get('crop_name', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Suitability Score:</strong> {crop.get('suitability_score', '-')}/10</p>
                        <p style="margin: 0.2rem 0;"><strong>Planting Season:</strong> {crop.get('planting_season', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Best Months to Grow:</strong> {crop.get('best_months_to_grow', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Estimated Profitability:</strong> <span style="background: rgba(77, 185, 128, 0.18); color: #e5f9e6; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: bold;">{crop.get('estimated_profitability', '-')}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div style="background: #162630; color: #edf2f7; padding: 1rem; border-radius: 12px; border: 1px solid rgba(77, 185, 128, 0.16);">
                        <h4 style="margin-top: 0; color: #f4f7fb;">💰 Financial Details</h4>
                        <p style="margin: 0.2rem 0;"><strong>Estimated Cost of Cultivation:</strong> {crop.get('estimated_cost_of_cultivation', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Expected Yield:</strong> {crop.get('expected_yield', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Expected Profit:</strong> <span style="color: #e5f9e6; font-weight: bold; font-size: 1.1rem;">{crop.get('expected_profit', '-')}</span></p>
                        <p style="margin: 0.2rem 0;"><strong>Seed Quantity Required:</strong> {crop.get('seed_quantity', '-')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background: #162630; color: #edf2f7; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(77, 185, 128, 0.16);">
                        <h4 style="margin-top: 0; color: #f4f7fb;">🌍 Growing Requirements</h4>
                        <p style="margin: 0.2rem 0;"><strong>Water Requirement:</strong> {crop.get('water_requirement', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Fertilizer Recommendation:</strong> {crop.get('fertilizer_recommendation', '-')}</p>
                        <p style="margin: 0.2rem 0;"><strong>Reason:</strong> {crop.get('reason', '-')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Seed Quantity Summary
    if seed_qty:
        st.markdown(
            f"""
            <div class='result-card'>
                <h3>📦 Seed Quantity Required</h3>
                <div class='metric-row'>
                    {''.join(f"<div class='mini-stat'><span>{k.replace('_',' ').title()}</span><strong>{v}</strong></div>" for k, v in seed_qty.items())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Farming Practices and Risks
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class='panel'>
                <h3>🧑‍🌾 Farming Practices</h3>
                <ul class='list-box'>
                    {''.join(f"<li>{p}</li>" for p in practices) if practices else "<li>No practices available</li>"}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='panel'>
                <h3>⚠️ Possible Risks</h3>
                <ul class='list-box'>
                    {''.join(f"<li>{r}</li>" for r in risks) if risks else "<li>No risks identified</li>"}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def predict(uploaded_image, location_value, weather_value, temperature_value, humidity_value, rainfall_value, land_size_value, budget_value):
    if not model_loaded:
        st.error("Prediction model is not currently available in this environment.")
        return
    
    if uploaded_image is None:
        st.error("Please upload your farm image first.")
        return
    
    if not location_value or not humidity_value or not rainfall_value or not land_size_value or not budget_value:
        st.error("Please provide all required farm parameters.")
        return

    image = Image.open(uploaded_image).convert("RGB")
    resized_image = image.resize((225, 225))

    image_arr = np.array(resized_image) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)

    prediction = model.predict(image_arr)

    class_name = {0: "Alluvial Soil", 1: "Black Soil", 2: "Clay Soil", 3: "Red Soil"}
    predicted_soil = class_name[np.argmax(prediction)]
    st.success(f"🎯 Predicted soil: {predicted_soil}")

    with st.spinner("🔄 Generating AI analysis and recommendations..."):
        response = askai(
            image=uploaded_image,
            soil_type=predicted_soil,
            location=location_value,
            weather=weather_value,
            temperature=temperature_value,
            humidity=humidity_value,
            rainfall=rainfall_value,
            land_size=land_size_value,
            budget=budget_value,
        )
        parsed = parse_ai_response(response)
        render_results(parsed, predicted_soil)


render_hero()

with st.form("soil_form"):
    col1, col2 = st.columns([1.05, 0.95], gap="large")
    
    with col1:
        st.markdown("<h3 style='color: #163426; margin-top: 0;'>📸 1. Upload Soil Image</h3>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Choose an image of your soil...", type=["jpg", "jpeg", "png"], key="soil_image")
        if uploaded_image:
            st.image(uploaded_image, caption="Selected Soil Image", use_container_width=True)
    
    with col2:
        st.markdown("<h3 style='color: #163426; margin-top: 0;'>📋 2. Farm Parameters</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6a7d6b; font-size: 0.9rem;'>Soil Type Override (Optional - leaves to AI model if blank):</p>", unsafe_allow_html=True)
        soil_override = st.text_input("Soil Type (optional)", placeholder="e.g., Black Soil, Clay Soil", label_visibility='collapsed')
        location_value = st.text_input("Location (District, State)", placeholder="e.g., Indore, MP India")
        weather_value = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Windy", "Humid", "Warm and Dusty"])
        temperature_value = st.number_input("Temperature (°C)", value=28.0, min_value=-40.0, max_value=60.0)
        humidity_value = st.text_input("Humidity (%)", placeholder="e.g., 65%")
        rainfall_value = st.text_input("Rainfall", placeholder="e.g., 700 mm")
        land_size_value = st.text_input("Land Size", placeholder="e.g., 2 acres")
        budget_value = st.number_input("Available Budget (INR)", value=50000, min_value=0)
    
    col1, col2 = st.columns(2)
    with col2:
        st.markdown("<p></p>", unsafe_allow_html=True)  # Spacing
    
    submitted = st.form_submit_button("🌍 Analyze Soil & Generate Report", use_container_width=True)

if submitted:
    predict(uploaded_image, location_value, weather_value, temperature_value, humidity_value, rainfall_value, land_size_value, budget_value)
