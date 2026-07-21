# SoilWise AI - Technical Documentation

**Version:** 1.0.0  
**Last Updated:** July 21, 2026  
**Status:** Production Ready  
**Technology Stack:** Python 3.9+, Streamlit, TensorFlow/Keras 2.13+, OpenRouter API

---

## 1. Executive Summary

**SoilWise AI** is a production-grade web application that combines computer vision and generative AI to deliver data-driven agricultural recommendations. The system employs a convolutional neural network (CNN) for soil type classification from image inputs, coupled with a large language model (LLM) for contextual crop recommendations and farming guidance based on environmental parameters.

### Key Capabilities
- **Soil Classification:** Real-time soil type prediction from photographic input using a pre-trained CNN model
- **Intelligent Recommendations:** Context-aware crop suggestions considering soil composition, local climate, and economic factors
- **Risk Assessment:** Automated identification of agricultural hazards and mitigation strategies
- **Decision Support:** Comprehensive farming practices and financial projections for informed decision-making

### Target Audience
- Small to medium-scale farmers
- Agricultural consultants and extension services
- Agribusiness enterprises
- Agricultural research institutions

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer (Streamlit)                │
│              (User Interface, Form Handling, Display)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐      ┌──────────▼─────────┐
│   ML Pipeline    │      │  API Integration    │
│   (Computer      │      │  Layer              │
│    Vision)       │      │  (OpenRouter)       │
│                  │      │                     │
│ - Image Load     │      │ - Request Builder   │
│ - Preprocessing  │      │ - Response Parser   │
│ - CNN Inference  │      │ - Error Handling    │
└────────┬─────────┘      └──────────┬──────────┘
         │                           │
    ┌────▼─────────────────────────────▼──┐
    │    Data Processing & Analysis       │
    │  (JSON Parsing, List Normalization) │
    └────┬──────────────────────────────┬──┘
         │                              │
    ┌────▼────────┐           ┌────────▼────┐
    │  Local Model │           │  Remote LLM │
    │  (TensorFlow)│           │  (OpenRouter)
    │  SoilModel   │           │  Nemotron   │
    │  (H5)        │           │             │
    └──────────────┘           └─────────────┘
```

### 2.2 Data Flow

```
User Input
   ├── Image Upload
   ├── Farm Parameters
   └── Environmental Data
         ↓
   Image Preprocessing
   ├── RGB Conversion
   ├── Resize (225×225)
   └── Normalization (0-1)
         ↓
   CNN Inference
   ├── Forward Pass
   ├── Class Prediction
   └── Soil Type Mapping
         ↓
   LLM Prompt Construction
   ├── Soil Characteristics
   ├── Environmental Context
   └── Economic Factors
         ↓
   OpenRouter API Call
   ├── Structured Prompt
   └── JSON Schema Compliance
         ↓
   Response Parsing
   ├── JSON Extraction
   ├── Error Handling
   └── Data Validation
         ↓
   Results Rendering
   ├── Soil Analysis Display
   ├── Crop Recommendations
   ├── Financial Projections
   └── Risk Assessment
```

---

## 3. Component Documentation

### 3.1 Frontend Module (`main.py`)

**Responsibility:** User interface, data input collection, and results visualization

**Key Functions:**

| Function | Purpose | Parameters | Returns |
|----------|---------|-----------|---------|
| `render_hero()` | Displays application header and feature overview | None | None (UI render) |
| `predict()` | Orchestrates prediction pipeline | image, location, weather, temp, humidity, rainfall, land_size, budget | Renders results |
| `render_results()` | Generates comprehensive results dashboard | result (dict), predicted_soil (str) | None (UI render) |
| `parse_ai_response()` | Extracts JSON from LLM response | raw_output (str/dict) | dict or None |
| `ensure_list()` | Normalizes list/string inputs for consistent rendering | value (any) | list |

**Configuration:**
```python
Model Path: Model/SoilModel.h5
Input Dimensions: 225 × 225 × 3 (RGB)
Output Classes: 4 (Alluvial, Black, Clay, Red)
Image Normalization: Divide by 255.0
Framework: TensorFlow/Keras
```

**State Management:**
- `model_loaded` (bool): Tracks successful model initialization
- `uploaded_image` (UploadedFile): Cached user image input
- `session_state`: Streamlit session persistence

### 3.2 AI Integration Module (`AskAi/Askai.py`)

**Responsibility:** LLM integration and crop recommendation generation

**Core Function:** `askai(**kwargs)`

**Input Parameters:**
```python
{
    'image': <UploadedFile>,          # User-provided soil image
    'soil_type': str,                 # CNN prediction output
    'location': str,                  # Geographic location (district, state)
    'weather': str,                   # Current weather condition
    'temperature': float,             # Celsius
    'humidity': float,                # Percentage (0-100)
    'rainfall': str,                  # Millimeters (formatted string)
    'land_size': str,                 # Area measurement (e.g., acres, hectares)
    'budget': float                   # INR (Indian Rupees)
}
```

**Output Schema:**
```json
{
  "soil_analysis": {
    "soil_type": "string",
    "soil_condition": "string",
    "moisture_level": "string",
    "fertility_rating": number (0-10),
    "weather_suitability": "string",
    "key_observations": {
      "texture": "string",
      "color": "string",
      "compactness": "string",
      "drainage": "string"
    },
    "suitability": "string"
  },
  "recommended_crops": [
    {
      "crop_name": "string",
      "suitability_score": number (0-10),
      "reason": "string",
      "planting_season": "string",
      "best_months_to_grow": "string",
      "water_requirement": "string",
      "fertilizer_recommendation": "string",
      "estimated_cost_of_cultivation": "currency string",
      "expected_yield": "string",
      "expected_profit": "currency string",
      "estimated_profitability": "string (High/Medium/Low)",
      "seed_quantity": "string"
    }
  ],
  "farming_practices": ["string"],
  "possible_risks": ["string"],
  "seed_quantity_required": { "crop_name": "quantity" },
  "most_profitable_crop": "string"
}
```

**API Configuration:**
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Model:** `nvidia/nemotron-3-nano-30b-a3b:free`
- **Authentication:** Bearer token (API key from `.env`)
- **Response Format:** JSON (enforced via prompt engineering)
- **Rate Limiting:** Subject to OpenRouter free tier constraints

**Error Handling:**
```python
HTTP Status Codes:
- 200: Success
- 429: Rate limited (retry with backoff)
- 401: Authentication failed (verify API key)
- 500: Server error (retry operation)
```

### 3.3 Styling Module (`styles.css`)

**Responsibility:** Visual presentation and responsive design

**Key Design Systems:**
- **Color Palette:** Dark theme with green accents (#4db980, #78d29c)
- **Typography:** Progressive scale (0.8rem → 2.25rem)
- **Spacing:** 0.45rem–2rem increments
- **Animations:** Fade-up (0.7–0.8s), Float loop (8s)
- **Responsiveness:** Single-column layout on screens ≤900px

---

## 4. Deployment & Setup

### 4.1 Prerequisites

| Component | Requirement | Version |
|-----------|-------------|---------|
| Python | Runtime | 3.9+ |
| pip | Package manager | Latest |
| Virtual Environment | Isolation | venv |
| API Key | OpenRouter access | Active subscription |
| Model File | Pre-trained CNN | SoilModel.h5 |

### 4.2 Installation

```bash
# Clone repository and navigate
cd "NN Soil Project"

# Create isolated Python environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies with pinned versions
pip install -r requirements.txt

# Verify installations
python -c "import tensorflow; import streamlit; print('✓ All dependencies installed')"
```

### 4.3 Configuration

**Create `.env` file in project root:**
```
Api_key=YOUR_OPENROUTER_API_KEY_HERE
```

**Verify model placement:**
```
NN Soil Project/
└── Model/
    └── SoilModel.h5  ✓ Required
```

### 4.4 Running the Application

```bash
# Start Streamlit server
streamlit run main.py

# Application launches at http://localhost:8501
# Accessible via:
# - Local: http://127.0.0.1:8501
# - Network: http://<your-ip>:8501
```

### 4.5 Production Considerations

**Current Limitation:** Application uses absolute path for model loading
```python
# Current (NOT portable):
model = models.load_model("C:\\Users\\win\\OneDrive\\Desktop\\NN Soil Project\\Model\\SoilModel.h5")

# Recommended (portable):
import os
model_path = os.path.join(os.path.dirname(__file__), 'Model', 'SoilModel.h5')
model = models.load_model(model_path)
```

---

## 5. Data Specifications

### 5.1 Soil Classification System

| Class ID | Soil Type | Key Characteristics | pH Range | Fertility |
|----------|-----------|-------------------|----------|-----------|
| 0 | Alluvial Soil | Silt/sand mix, light-colored, good drainage | 7.0-8.5 | High |
| 1 | Black Soil | Clay-rich, dark, water-retentive | 7.5-8.5 | High |
| 2 | Clay Soil | High clay content, compact, poor drainage | 6.0-7.5 | Medium |
| 3 | Red Soil | Iron oxide-rich, porous, acidic | 5.0-7.0 | Medium |

### 5.2 Image Input Specifications

| Parameter | Specification |
|-----------|---------------|
| **Format** | JPEG, PNG |
| **Input Size** | Any resolution |
| **Processing Size** | 225 × 225 pixels |
| **Color Space** | RGB (converted if needed) |
| **Normalization** | Pixel values: 0.0–1.0 (divide by 255) |
| **Max Upload Size** | Browser dependent (typically 200MB) |

### 5.3 Environmental Parameters Schema

```python
location: str              # Format: "District, State, Country"
weather: str              # Enum: [Sunny, Cloudy, Rainy, Windy, Humid, Warm and Dusty]
temperature: float        # Range: -40°C to 60°C
humidity: float           # Range: 0-100 (percentage)
rainfall: str             # Format: "numeric mm" or "numeric inches"
land_size: str            # Format: "numeric acres/hectares/sq_m"
budget: float             # Currency in INR (₹)
```

---

## 6. Error Handling & Resilience

### 6.1 Model Loading Failures

**Scenario:** TensorFlow/Keras unavailable or model file corrupted

**Mitigation:**
```python
try:
    from tensorflow.keras import models
except Exception:
    models = None  # Graceful degradation

if models is None:
    st.warning("TensorFlow not installed. Prediction unavailable.")
```

**User Impact:** Predictions disabled; API recommendations still available

### 6.2 API Communication Failures

**Scenario:** OpenRouter endpoint unreachable, rate-limited, or unauthorized

**Mitigation:**
- Verify API key in `.env`
- Check OpenRouter service status
- Implement exponential backoff for retries
- Display user-friendly error messages
- Cache recent responses when feasible

**Recovery Time:** Depends on rate limit window (typically 60 seconds)

### 6.3 JSON Response Parsing

**Scenario:** LLM returns malformed JSON or incomplete data

**Mitigation:**
```python
def parse_ai_response(raw_output):
    # Attempt direct parsing
    # Fallback to substring extraction
    # Return None if unrecoverable
```

**Fallback Behavior:** Display partial results or generic error message

---

## 7. Performance Optimization

### 7.1 Execution Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| App Startup | 2–3 sec | Model loading (one-time) |
| Image Upload | < 1 sec | Browser operation |
| Preprocessing | 0.5 sec | Resize, normalize |
| CNN Inference | 1–2 sec | CPU/GPU dependent |
| LLM Request | 5–10 sec | Network-bound |
| Response Parsing | 0.5 sec | JSON parsing |
| UI Rendering | 1–2 sec | Streamlit re-run |
| **Total** | **~10–20 sec** | Typical end-to-end |

### 7.2 Optimization Opportunities

| Opportunity | Implementation | Impact |
|-------------|-----------------|--------|
| Model Caching | Load once, reuse | -2–3 sec startup |
| Response Caching | Store recent predictions | -5–10 sec repeat |
| Async API Calls | Non-blocking requests | Improved UX |
| Image Optimization | Compress before upload | -1–2 sec transfer |
| Progressive Rendering | Stream results as available | Better UX |

### 7.3 Scalability Considerations

**Current Constraints:**
- Single-threaded Streamlit sessions
- Free-tier API rate limits (~60 req/min)
- Local model inference (CPU-bound)

**Path to Production Scale:**
1. Containerize with Docker
2. Deploy on Kubernetes or serverless platform
3. Implement Redis caching layer
4. Use GPU acceleration for inference
5. Establish LLM provider partnership
6. Implement request queuing

---

## 8. Security & Best Practices

### 8.1 API Key Management

**Threat:** Hardcoded credentials in source code

**Mitigation:**
- ✓ Store keys in `.env` file (excluded from Git)
- ✓ Use environment variables via `python-dotenv`
- ✓ Rotate keys regularly (recommended monthly)
- ✓ Implement key versioning if multiple deployments
- ✓ Monitor API usage for unusual patterns

### 8.2 Input Validation

**Threat:** Malicious file uploads, invalid parameter ranges

**Current Implementation:**
```python
st.file_uploader(..., type=["jpg", "jpeg", "png"])  # File type restriction
st.number_input(..., min_value=-40.0, max_value=60.0)  # Range validation
```

**Recommended Enhancements:**
- File size limits (max 5MB per image)
- MIME type verification
- Image dimension validation pre-processing
- Sanitization of string inputs

### 8.3 Data Privacy

**Considerations:**
- Images processed locally (not stored)
- Environmental parameters sent to OpenRouter (third-party)
- No user data persistence
- Compliance with GDPR/CCPA (if applicable)

**Recommendations:**
- Implement privacy policy documentation
- Provide data processing disclosure to users
- Establish data retention/deletion policies

---

## 9. Maintenance & Operations

### 9.1 Common Issues & Resolution

| Issue | Root Cause | Resolution |
|-------|-----------|-----------|
| Model not found | Path mismatch | Verify file location, use relative paths |
| API 401 error | Invalid/expired key | Update `.env` with valid API key |
| API 429 error | Rate limit exceeded | Wait 60 seconds, implement retry logic |
| Image processing error | Corrupted file | Regenerate or re-upload image |
| Partial results | JSON parsing failure | Check API response format, add logging |

### 9.2 Monitoring Recommendations

**Metrics to Track:**
- API response times (target: <10 sec)
- Error rates (target: <1%)
- Model inference latency (target: <2 sec)
- User session duration
- Feature utilization rates

**Logging Implementation:**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Prediction: {predicted_soil} (confidence: {max_prob:.2%})")
logger.error(f"API Error: {status_code} - {error_message}")
```

### 9.3 Dependency Management

**Regular Maintenance Tasks:**
- Monthly: Review and update dependencies
- Quarterly: Security audit of packages
- Semi-annually: Major version compatibility checks

**Critical Dependencies:**
```
TensorFlow 2.13+     → Model inference engine
Streamlit 1.28+      → Web framework
Requests 2.31+       → HTTP library
python-dotenv 1.0+   → Environment configuration
```

---

## 10. Future Roadmap

### Phase 2 (Q3 2026)
- Multi-image batch analysis
- Export recommendations as PDF/Excel
- User account system with history tracking
- Improved LLM model selection

### Phase 3 (Q4 2026)
- Mobile-responsive UI enhancement
- Real-time weather API integration
- Crop market price predictions
- Offline inference capability

### Phase 4 (2027)
- Native mobile applications (iOS/Android)
- Multi-language support (Hindi, Marathi, Tamil)
- Community knowledge sharing platform
- Integration with agricultural supply chains

---

## 11. Support & Contributing

**Issue Reporting:**
- Document symptoms and steps to reproduce
- Include logs from `.venv/` or terminal
- Provide system information (Python version, OS)

**Development Workflow:**
```bash
# Create feature branch
git checkout -b feature/description

# Commit changes
git commit -m "Brief description"

# Push and create pull request
git push origin feature/description
```

---

## 12. Appendix

### A. Directory Structure Explained

```
NN Soil Project/
│
├── main.py                    # Entry point; UI and prediction orchestration
├── requirements.txt           # Pinned dependency versions
├── styles.css                 # Streamlit custom styling
├── .env                       # Secrets (excluded from Git)
├── .gitignore                 # Git exclusion rules
│
├── AskAi/                     # LLM integration module
│   ├── __init__.py
│   └── Askai.py              # OpenRouter API wrapper
│
├── Model/                     # Pre-trained models
│   └── SoilModel.h5          # CNN classifier (TensorFlow format)
│
├── Data/                      # Training dataset reference
│   ├── Alluvial soil/
│   ├── Black Soil/
│   ├── Clay soil/
│   └── Red soil/
│
├── Training Soil Data/        # Development artifacts
│   └── trainSoilData.ipynb   # Jupyter notebook for model training
│
└── .venv/                     # Python virtual environment (local)
```

### B. Quick Reference: Key Commands

```bash
# Activate environment
.venv\Scripts\activate

# Install/update dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py

# Run syntax check
python -m py_compile main.py

# View API documentation
# → https://openrouter.ai/docs
```

---

**Document Version:** 1.0.0  
**Last Reviewed:** July 21, 2026  
**Maintained By:** Development Team  
**License:** Confidential – All Rights Reserved
