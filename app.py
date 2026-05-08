"""
FloodSense — District Official Dashboard
=========================================
AI-powered flood early warning system for Pakistani districts.

Run:
    streamlit run app.py

Brief requirements satisfied:
  - No ML jargon visible to user
  - Inputs: rainfall, date, district, soil condition, visible surface water
  - Output: colour-coded risk badge, confidence, population, action
  - Bilingual: English + Urdu throughout
  - Handles missing inputs gracefully (no crashes)
  - Page <2MB and loads in <10s on slow internet (no external assets except 1 Google Font)
  - Rule-based safety overrides for cloudburst-level rainfall
"""

from datetime import date as date_type
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="FloodSense — Flood Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CUSTOM CSS — injected via st.markdown to give the page a polished look.
# This is the "frontend code" — pure HTML/CSS, no external assets except
# a single small Google Font (~30KB, browser-cached after first load).
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Nastaliq+Urdu&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f9ff 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* === HERO HEADER === */
.hero {
    background: linear-gradient(135deg, #0c4a6e 0%, #075985 50%, #0369a1 100%);
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 10px 30px rgba(7, 89, 133, 0.25);
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1;
}
.hero-subtitle {
    font-size: 16px;
    margin-top: 6px;
    opacity: 0.85;
    font-weight: 400;
}
.hero-urdu {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 18px;
    margin-top: 4px;
    opacity: 0.95;
}
.hero-tagline {
    font-size: 13px;
    margin-top: 14px;
    padding: 6px 12px;
    background: rgba(255,255,255,0.15);
    display: inline-block;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 600;
}

/* === INPUT CARD === */
.input-card {
    background: white;
    padding: 28px 32px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    margin-bottom: 24px;
    border: 1px solid rgba(15, 23, 42, 0.05);
}
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.stSelectbox label, .stDateInput label, .stNumberInput label {
    font-weight: 600 !important;
    color: #1e293b !important;
    font-size: 14px !important;
}

/* === PRIMARY BUTTON === */
.stButton > button {
    background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 24px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(3, 105, 161, 0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(3, 105, 161, 0.4) !important;
}

/* === RISK BADGE === */
.risk-badge {
    padding: 36px 28px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 24px 0;
    box-shadow: 0 12px 32px rgba(0,0,0,0.15);
    animation: fadeInUp 0.5s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.risk-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 4px;
    opacity: 0.9;
    margin-bottom: 8px;
}
.risk-level-en {
    font-size: 64px;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0;
    letter-spacing: -0.02em;
}
.risk-level-ur {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 32px;
    margin-top: 8px;
    opacity: 0.95;
}

/* === METRIC CARDS === */
.metric-card {
    background: white;
    padding: 20px 24px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    border: 1px solid rgba(15, 23, 42, 0.05);
    text-align: center;
    height: 100%;
}
.metric-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #0c4a6e;
    line-height: 1;
}
.metric-urdu {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 13px;
    color: #94a3b8;
    margin-top: 6px;
}

/* === ACTION CARD === */
.action-card {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 24px 28px;
    border-radius: 14px;
    border-left: 5px solid #f59e0b;
    margin: 20px 0;
}
.action-title {
    font-size: 13px;
    font-weight: 700;
    color: #78350f;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.action-text-en {
    font-size: 17px;
    color: #1e293b;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 10px;
}
.action-text-ur {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 18px;
    color: #1e293b;
    line-height: 1.8;
    direction: rtl;
    text-align: right;
}

/* === SAFEGUARD NOTICE === */
.safeguard-notice {
    background: #ecfdf5;
    border: 1.5px solid #6ee7b7;
    padding: 16px 20px;
    border-radius: 12px;
    margin-top: 16px;
    font-size: 14px;
    color: #064e3b;
}
.safeguard-notice strong {
    display: block;
    margin-bottom: 6px;
    color: #047857;
}

.warning-card {
    background: #fff7ed;
    border-left: 5px solid #ea580c;
    padding: 18px 22px;
    border-radius: 12px;
    margin: 16px 0;
    color: #7c2d12;
    font-weight: 500;
}

.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    line-height: 1.7;
}

@media (max-width: 768px) {
    .hero-title { font-size: 28px; }
    .risk-level-en { font-size: 48px; }
    .metric-value { font-size: 24px; }
    .block-container { padding-top: 1rem; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# DATA + MODEL LOADING (cached so the page loads fast)
# ============================================================================
ART = Path("artifacts")
DATA = Path("data")  # CSVs live in ./data/ inside the repo


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.pkl"), joblib.load(ART / "feature_cols.pkl")


@st.cache_data
def load_history():
    df = pd.read_csv(DATA / "floodsense_training_data.csv")
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df = df[df["precipitation"] != -999]
    df = df[df["temperature"] != -999]
    df = df[df["soil_moisture"] <= 1.0]
    df = df[(df["humidity"] <= 100) & (df["humidity"] >= 0)]
    elev = pd.read_csv(DATA / "district_elevation_reference.csv")
    df = df.merge(elev, on="district", how="left")
    return df


DISTRICT_POP = {
    "Sindh_District": 1_400_000,
    "Balochistan_District": 850_000,
    "KP_District": 1_900_000,
}

SOIL_MAP = {
    "Dry / خشک": 0.10,
    "Moist / نم": 0.30,
    "Saturated / سیراب": 0.55,
}

RISK_BANDS = [
    (0.00, 0.25, "LOW", "خطرہ کم", "#16a34a",
     "Continue normal operations. Monitor weather updates twice daily.",
     "معمول کے کام جاری رکھیں۔ موسم کی اپ ڈیٹس روزانہ دو بار دیکھیں۔"),
    (0.25, 0.50, "MEDIUM", "خطرہ درمیانہ", "#eab308",
     "Alert local response teams. Check drainage and inform vulnerable households.",
     "مقامی ٹیموں کو الرٹ کریں۔ نکاسی چیک کریں اور کمزور گھرانوں کو مطلع کریں۔"),
    (0.50, 0.75, "HIGH", "خطرہ زیادہ", "#f97316",
     "Pre-position relief supplies. Begin voluntary evacuation of low-lying areas.",
     "امدادی سامان پہلے سے رکھیں۔ نشیبی علاقوں سے رضاکارانہ انخلا شروع کریں۔"),
    (0.75, 1.01, "CRITICAL", "خطرہ شدید", "#dc2626",
     "EVACUATE NOW. Mobilise rescue. Open relief camps. Cut power to flood zones.",
     "ابھی انخلا کریں۔ ریسکیو کو متحرک کریں۔ امدادی کیمپ کھولیں۔ سیلابی علاقوں کی بجلی بند کریں۔"),
]


def risk_band(prob: float):
    for lo, hi, en, ur, color, act_en, act_ur in RISK_BANDS:
        if lo <= prob < hi:
            return en, ur, color, act_en, act_ur
    return RISK_BANDS[-1][2:]


def apply_safeguards(prob, rainfall_mm, soil_value, visible_water):
    notes = []
    floor = 0.0
    if rainfall_mm >= 100:
        floor = max(floor, 0.78)
        notes.append("Extreme rainfall (≥100mm/24h) — auto-elevated to Critical")
    elif rainfall_mm >= 50:
        floor = max(floor, 0.55)
        notes.append("Heavy rainfall (≥50mm/24h) — auto-elevated to High")
    elif rainfall_mm >= 25 and soil_value >= 0.5:
        floor = max(floor, 0.30)
        notes.append("Moderate rain on saturated soil — auto-elevated to Medium")
    if visible_water and rainfall_mm >= 25:
        floor = max(floor, 0.55)
        notes.append("Visible surface water + recent rainfall — escalated")
    return max(prob, floor), notes


def build_features(rainfall_mm, date_val, district, soil_value, visible_water,
                   history_df, feature_cols):
    month = date_val.month
    doy = date_val.timetuple().tm_yday

    hist = history_df[(history_df["district"] == district) &
                      (history_df["month"] == month)]
    if len(hist) < 3:
        hist = history_df[history_df["district"] == district]

    def med(col, fallback=0.0):
        v = hist[col].median() if col in hist.columns else fallback
        return fallback if pd.isna(v) else v

    train_precip = history_df["precipitation"].dropna()
    train_precip = train_precip[train_precip > 0].sort_values()
    if rainfall_mm <= 0:
        precip_scaled = 0.0
    elif rainfall_mm < 5:
        precip_scaled = train_precip.quantile(0.50)
    elif rainfall_mm < 25:
        precip_scaled = train_precip.quantile(0.80)
    elif rainfall_mm < 75:
        precip_scaled = train_precip.quantile(0.95)
    elif rainfall_mm < 150:
        precip_scaled = train_precip.quantile(0.99)
    else:
        precip_scaled = float(train_precip.max())

    effective_soil = soil_value
    if rainfall_mm >= 75:
        effective_soil = max(effective_soil, 0.55)
    if rainfall_mm >= 150 or visible_water:
        effective_soil = max(effective_soil, 0.65)

    row = {c: 0.0 for c in feature_cols}
    row["precipitation"] = precip_scaled
    row["soil_moisture"] = effective_soil
    row["month"] = month
    row["day_of_year"] = doy
    row["is_monsoon"] = 1 if month in [6, 7, 8, 9] else 0

    for col in ["evaporation", "pressure", "temperature", "wind_speed",
                "humidity", "temp_3day_avg", "avg_elevation_m"]:
        if col in row:
            row[col] = med(col)

    base_3day = med("precip_3day_avg")
    base_7day = med("precip_7day_avg")
    row["precip_3day_avg"] = max(base_3day, precip_scaled * 0.6)
    row["precip_7day_avg"] = max(base_7day, precip_scaled * 0.4)
    row["soil_3day_avg"] = max(med("soil_3day_avg"), effective_soil * 0.9)

    for lag in [1, 2, 3]:
        if f"precip_lag_{lag}" in row:
            row[f"precip_lag_{lag}"] = precip_scaled * (0.7 ** lag)
        if f"soil_lag_{lag}" in row:
            row[f"soil_lag_{lag}"] = effective_soil
    if "precip_roll_5" in row:
        row["precip_roll_5"] = precip_scaled * 3
    if "precip_roll_14" in row:
        row["precip_roll_14"] = precip_scaled * 5

    if "saturation_index" in row:
        row["saturation_index"] = effective_soil * row["precip_3day_avg"]
    if "water_per_meter" in row:
        row["water_per_meter"] = 0

    for c in feature_cols:
        if c.startswith("district_"):
            row[c] = 1.0 if c == f"district_{district}" else 0.0
        if c.startswith("terrain_type_"):
            terrain = hist["terrain_type"].iloc[0] if len(hist) else ""
            row[c] = 1.0 if c == f"terrain_type_{terrain}" else 0.0

    return pd.DataFrame([row])[feature_cols]


# ============================================================================
# UI
# ============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌊 FloodSense</div>
    <div class="hero-subtitle">AI-powered flood early warning for Pakistani districts</div>
    <div class="hero-urdu">پاکستانی اضلاع کے لیے سیلاب کا ابتدائی انتباہ</div>
    <div class="hero-tagline">Built for Better Tomorrow · BNU Tech Fest 2026</div>
</div>
""", unsafe_allow_html=True)

try:
    model, feature_cols = load_model()
    history_df = load_history()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Run the training notebook first to create `artifacts/model.pkl`.")
    st.stop()

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Site Conditions · حالات کا اندراج</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    district = st.selectbox(
        "District / ضلع",
        options=sorted(history_df["district"].unique()),
    )
    selected_date = st.date_input(
        "Date / تاریخ",
        value=date_type(2025, 8, 15),
    )
    rainfall = st.number_input(
        "Rainfall last 24 hours (mm) / گزشتہ 24 گھنٹوں کی بارش",
        min_value=0.0, max_value=500.0, value=0.0, step=1.0,
    )

with col2:
    soil = st.selectbox(
        "Soil Condition / زمین کی حالت",
        options=list(SOIL_MAP.keys()),
        index=1,
    )
    visible_water = st.selectbox(
        "Visible surface water? / نظر آنے والا پانی؟",
        options=["No / نہیں", "Yes / ہاں"],
    )
    visible_water_bool = visible_water.startswith("Yes")
    st.write("")

st.markdown('</div>', unsafe_allow_html=True)

go = st.button("🔍 Assess Flood Risk · سیلاب کا خطرہ جانچیں",
               use_container_width=True)

if go:
    if district is None or rainfall is None:
        st.markdown(
            '<div class="warning-card">⚠️ <strong>Insufficient data — manual assessment recommended.</strong>'
            '<br/>ناکافی معلومات — دستی تشخیص کی سفارش کی جاتی ہے۔</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        X = build_features(
            rainfall_mm=rainfall, date_val=selected_date,
            district=district, soil_value=SOIL_MAP[soil],
            visible_water=visible_water_bool,
            history_df=history_df, feature_cols=feature_cols,
        )
        prob_raw = float(model.predict_proba(X)[0, 1])
        prob, safeguard_notes = apply_safeguards(
            prob_raw, rainfall, SOIL_MAP[soil], visible_water_bool
        )
        en, ur, color, act_en, act_ur = risk_band(prob)
        confidence = int(round(max(prob, 1 - prob) * 100))
        pop = DISTRICT_POP.get(district, 0)
        pop_at_risk = int(pop * min(prob * 1.5, 1.0))

        st.markdown(f"""
        <div class="risk-badge" style="background: linear-gradient(135deg, {color}dd 0%, {color} 100%);">
            <div class="risk-label">⚠️ FLOOD RISK · سیلاب کا خطرہ</div>
            <div class="risk-level-en">{en}</div>
            <div class="risk-level-ur">{ur}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence}%</div>
                <div class="metric-urdu">اعتماد</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">People at Risk</div>
                <div class="metric-value">{pop_at_risk:,}</div>
                <div class="metric-urdu">متاثرہ آبادی</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{int(prob*100)}%</div>
                <div class="metric-urdu">خطرے کا اسکور</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">📋 Recommended Action · تجویز کردہ اقدام</div>
            <div class="action-text-en">{act_en}</div>
            <div class="action-text-ur">{act_ur}</div>
        </div>
        """, unsafe_allow_html=True)

        if safeguard_notes:
            notes_html = "".join([f"<li>{n}</li>" for n in safeguard_notes])
            st.markdown(f"""
            <div class="safeguard-notice">
                <strong>🛡️ Safety override applied</strong>
                <ul style="margin: 4px 0 8px 18px; padding: 0;">{notes_html}</ul>
                <span style="font-size: 12px; opacity: 0.8;">
                    Model raw score: {int(prob_raw*100)}% → After safety rules: {int(prob*100)}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        if 0.40 < prob_raw < 0.60 and not safeguard_notes:
            st.markdown(
                '<div class="warning-card">⚠️ <strong>Borderline result — manual assessment recommended.</strong>'
                '<br/>نتیجہ غیر یقینی — دستی تشخیص کی سفارش۔</div>',
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.markdown(
            f'<div class="warning-card">Could not assess risk: {e}<br/>Please check inputs.</div>',
            unsafe_allow_html=True,
        )

st.markdown("""
<div class="app-footer">
    Trained on 2022–2024 flood records across Sindh, Balochistan & KP.<br/>
    Always corroborate with on-ground observations. Not a replacement for official PMD/NDMA bulletins.<br/>
    <span style="opacity: 0.7;">© Neural Nova FloodSense · Built for Better Tomorrow</span>
</div>
""", unsafe_allow_html=True)
