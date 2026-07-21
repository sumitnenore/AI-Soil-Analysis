import requests
import os
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

API_KEY = st.secrets["OpenRouter_Api_Key"]

# API_KEY = os.getenv("Api_key")


def askai(**kwargs):
    example_json = """{
  "soil_analysis": {
    "soil_type": "Black Soil",
    "soil_condition": "Highly fertile with good moisture retention and clay texture.",
    "moisture_level": "High",
    "fertility_rating": 8.8,
    "weather_suitability": "Good",
    "key_observations": {
      "texture": "Clayey",
      "color": "Dark Black",
      "compactness": "Medium",
      "drainage": "Moderate"
    },
    "suitability": "Excellent"
  },
  "recommended_crops":
    {
      "crop_name": "Soybean",
      "suitability_score": 9.8,
      "reason": "Excellent compatibility with black soil and monsoon weather.",
      "planting_season": "Kharif",
      "best_months_to_grow": "June - September",
      "water_requirement": "Moderate",
      "fertilizer_recommendation": "DAP + Urea",
      "estimated_cost_of_cultivation": "₹32,000",
      "expected_yield": "18 Quintals",
      "expected_profit": "₹75,000",
      "estimated_profitability": "High",
      "seed_quantity": "35 kg"
    }
}"""

    prompt = f"""
You are an expert agricultural consultant.

The soil type has already been predicted by a machine learning model.

Input:

- Image : {kwargs.get('image')}
- Soil Type: {kwargs.get('soil_type')}
- Location: {kwargs.get('location')}
- Weather Condition: {kwargs.get('weather')}
- Temperature: {kwargs.get('temperature')}
- Humidity: {kwargs.get('humidity')}
- Rainfall: {kwargs.get('rainfall')}
- Land Size: {kwargs.get('land_size')}
- Budget: {kwargs.get('budget')}

Your task is to:

1. Explain the characteristics of the predicted soil type.
2. Analyze the soil condition and agricultural suitability.
3. Recommend the 5 best crops that can be grown in this soil under the given weather conditions.
4. For each crop provide:
   - suitability_score
   - reason
   - planting_season
   - best_months_to_grow
   - water_requirement
   - fertilizer_recommendation
   - estimated_cost_of_cultivation
   - expected_yield
   - expected_profit
   - estimated_profitability

5. Suggest farming practices.
6. Mention possible risks.
7. Estimate seed quantity required.
8. Recommend the most profitable crop.

IMPORTANT:
Return ONLY valid JSON.
Do NOT return markdown.
Do NOT return explanations outside JSON.

Example = {example_json}

"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            # "model": "openai/gpt-oss-120b:free",
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
    )

    if response.status_code == 200:
        result = response.json() 
        # return result
        return result["choices"][0]["message"]["content"]

    else:
        print("Error:", response.status_code)
        print(response.text)
        return None
