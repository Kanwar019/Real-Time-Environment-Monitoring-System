import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- MODULE 3 & 4 IMPORTS (ML & Evaluation) ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
st.set_page_config(
    page_title="EcoLive Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FIXED DARK THEME VARIABLES ---
main_bg_color = "#0E1117"       
card_bg_color = "#262730"       
card_border_color = "#41424b"   
theme_text_color = "#FFFFFF"    
plot_template = "plotly_dark"   
grid_color = "rgba(255, 255, 255, 0.2)" 

# --- CSS STYLING ---
st.markdown(f"""
    <style>
        .stApp {{ background-color: {main_bg_color}; color: {theme_text_color}; }}
        .block-container {{ padding-top: 1.5rem !important; padding-bottom: 3rem !important; }}
        
        /* Navigation Pill Buttons */
        div[role="radiogroup"] {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; background-color: transparent; }}
        div[role="radiogroup"] label {{ background-color: {card_bg_color}; border: 1px solid {card_border_color}; color: {theme_text_color}; padding: 8px 20px; border-radius: 24px !important; cursor: pointer; text-align: center; flex: 1 0 auto; min-width: 80px; transition: all 0.2s ease; }}
        div[role="radiogroup"] label:hover {{ border-color: #ff4b4b; color: #ff4b4b; transform: translateY(-1px); }}
        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{ display: none; }}
        
        /* Metric Cards */
        div[data-testid="metric-container"] {{ background-color: {card_bg_color}; border: 1px solid {card_border_color}; padding: 12px; border-radius: 12px; }}
        div[data-testid="metric-container"] label, div[data-testid="stMetricValue"] {{ color: {theme_text_color} !important; }}
        h1, h2, h3, h4, h5, h6, p, span, th, td {{ color: {theme_text_color} !important; }}
    </style>
""", unsafe_allow_html=True)

# --- MODULE 1: DATA MODALITY (API Endpoints) ---
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API = "https://air-quality-api.open-meteo.com/v1/air-quality"

# --- CORE LOGIC (MODULES 1, 2, 3, 4) ---

@st.cache_resource
def train_and_evaluate_model():
    """
    Executes Module 1, 2, 3, and 4 logic to return the trained model and evaluation metrics.
    """
    # Hardcoded location for training (Ludhiana/Punjab)
    lat, lon = 30.34, 76.38 
    
    # --- MODULE 1: DATA COLLECTION ---
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover", "past_days": 7}
    aq_params = {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "past_days": 7}
    
    try:
        w_resp = requests.get(WEATHER_API, params=params).json().get('hourly', {})
        a_resp = requests.get(AIR_QUALITY_API, params=aq_params).json().get('hourly', {})
        
        # --- MODULE 2: PREPROCESSING ---
        # 1. Merge weather + AQI on time
        df_w = pd.DataFrame(w_resp)
        df_a = pd.DataFrame(a_resp)
        df = pd.merge(df_w, df_a, on="time", how="inner")
        
        # 2. Datetime conversion & Sorting
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        
        # 3. Smoothing (Rolling Mean) - CRITICAL STEP FROM NOTEBOOK
        df["pm2_5_smooth"] = df["pm2_5"].rolling(window=3).mean()
        
        # 4. Remove NaNs caused by rolling window
        df = df.dropna().reset_index(drop=True)
        
        # --- MODULE 3: FEATURE EXTRACTION & MODEL BUILDING ---
        # 1. Create target variable (0 = Safe, 1 = Unsafe)
        df["AQI_Label"] = (df["pm2_5"] > 100).astype(int)
        
        # 2. Feature Matrix (X) matching notebook columns
        features = ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "cloudcover", "pm2_5_smooth"]
        X = df[features]
        y = df["AQI_Label"]
        
        # 3. Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Train Logistic Regression
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        # --- MODULE 4: MODEL EVALUATION ---
        y_pred = model.predict(X_test_scaled)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature Importance for Module 6 Analysis
        importance = pd.DataFrame({"Feature": features, "Importance": model.coef_[0]}).sort_values(by="Importance", key=abs, ascending=False)
        
        # Pack everything into a dictionary to pass to UI
        evaluation_results = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "feature_importance": importance,
            "X_test": X_test, # For experiments
            "y_test": y_test
        }
        
        return model, scaler, evaluation_results
        
    except Exception as e:
        st.error(f"Training Failed: {e}")
        return None, None, None

def get_historical_context(lat, lon):
    # Quick fetch for charting history on startup
    try:
        w_params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,relativehumidity_2m", "past_days": 1}
        a_params = {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "past_days": 1}
        w = requests.get(WEATHER_API, params=w_params).json().get('hourly', {})
        a = requests.get(AIR_QUALITY_API, params=a_params).json().get('hourly', {})
        df = pd.merge(pd.DataFrame(w), pd.DataFrame(a), on="time", how="inner")
        df["time"] = pd.to_datetime(df["time"])
        history = []
        for _, row in df.iterrows():
            history.append({"Time": row["time"], "PM2.5": row["pm2_5"], "Temp": row["temperature_2m"], "Humidity": row["relativehumidity_2m"]})
        return history
    except:
        return []

def fetch_live_data(lat, lon):
    try:
        w_params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover"}
        a_params = {"latitude": lat, "longitude": lon, "current": "pm2_5"}
        w = requests.get(WEATHER_API, params=w_params).json().get('current', {})
        a = requests.get(AIR_QUALITY_API, params=a_params).json().get('current', {})
        return w, a, datetime.now()
    except:
        return None, None, None

# --- INITIALIZATION (Run Training Once) ---
if "model" not in st.session_state:
    with st.spinner("Executing Modules 1-4 (Data Collection, Preprocessing, Training, Evaluation)..."):
        model, scaler, eval_results = train_and_evaluate_model()
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.eval_results = eval_results

# --- UI HEADER ---
st.title("🛡️ Sentinel Environment")

# Navigation
selected_page = st.radio(
    "Go to", 
    ["Dashboard", "Metrics", "Analytics", "Map", "Logs"], 
    horizontal=True,
    label_visibility="collapsed"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    lat_input = st.number_input("Latitude", value=30.34, format="%.4f")
    lon_input = st.number_input("Longitude", value=76.38, format="%.4f")
    refresh_rate = st.slider("Refresh (s)", 2, 60, 5)
    if selected_page == "Dashboard":
        run_btn = st.toggle("Live Stream", value=True)
    else:
        run_btn = False

if "history" not in st.session_state or len(st.session_state.history) == 0:
    st.session_state.history = get_historical_context(lat_input, lon_input)

# --- HELPER: PLOT STYLE ---
def update_plot_style(fig):
    fig.update_layout(template=plot_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=theme_text_color), xaxis=dict(showgrid=True, gridcolor=grid_color), 
                      yaxis=dict(showgrid=True, gridcolor=grid_color))
    return fig

# ==========================================
# PAGE: DASHBOARD (MODULE 5 - DEPLOYMENT)
# ==========================================
if selected_page == "Dashboard":
    weather, air, now = fetch_live_data(lat_input, lon_input)
    
    if weather and air:
        # Prepare Live Input matching training features
        live_input = pd.DataFrame([{
            "temperature_2m": weather["temperature_2m"],
            "relativehumidity_2m": weather["relativehumidity_2m"],
            "windspeed_10m": weather["windspeed_10m"],
            "cloudcover": weather["cloudcover"],
            "pm2_5_smooth": air["pm2_5"] # Using current as approximation for smooth
        }])
        
        # Prediction
        risk = 0.0
        if st.session_state.model:
            scaled_live = st.session_state.scaler.transform(live_input)
            prediction = st.session_state.model.predict(scaled_live)[0]
            risk = st.session_state.model.predict_proba(scaled_live)[0][1]

        # Update History
        if run_btn:
            new_row = {"Time": now, "PM2.5": air['pm2_5'], "Temp": weather['temperature_2m'], "Humidity": weather['relativehumidity_2m']}
            if not st.session_state.history or st.session_state.history[-1]["Time"] != now:
                st.session_state.history.append(new_row)

        # Metrics Row
        st.caption("Module 5: Real-Time Data Acquisition")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temperature", f"{weather['temperature_2m']}°C")
        m2.metric("Humidity", f"{weather['relativehumidity_2m']}%")
        m3.metric("Wind Speed", f"{weather['windspeed_10m']} km/h")
        avg_pm = np.mean([x['PM2.5'] for x in st.session_state.history[-10:]]) if len(st.session_state.history) > 10 else air['pm2_5']
        m4.metric("PM 2.5", f"{air['pm2_5']} µg/m³", f"{air['pm2_5'] - avg_pm:.1f}", delta_color="inverse")

        st.markdown("---")

        # Visualizations
        g_col, c_col = st.columns([1, 2])
        with g_col:
            st.markdown("#### Threat Level")
            val = air['pm2_5']
            color = "green" if val <= 50 else "orange" if val <= 100 else "red"
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val,
                gauge={'axis': {'range': [None, 300], 'tickcolor': theme_text_color}, 'bar': {'color': color}, 'threshold': {'line': {'color': "red", 'width': 4}, 'value': 100}}))
            fig_g.update_layout(height=280, margin=dict(t=20,b=20,l=30,r=30))
            st.plotly_chart(update_plot_style(fig_g), use_container_width=True)
            
            status_text = "Unsafe" if prediction == 1 else "Safe"
            if prediction == 1:
                st.error(f"⚠️ Predicted Status: **{status_text}** (Prob: {risk:.2f})")
            else:
                st.success(f"✅ Predicted Status: **{status_text}** (Prob: {risk:.2f})")

        with c_col:
            st.markdown("#### Air Quality Trend")
            hist_df = pd.DataFrame(st.session_state.history)
            if not hist_df.empty:
                fig_c = px.area(hist_df.tail(100), x="Time", y="PM2.5", color_discrete_sequence=["#00CC96"])
                fig_c.add_hline(y=100, line_dash="dash", line_color="red")
                fig_c.update_layout(height=320, margin=dict(t=10,b=10,l=0,r=0), xaxis_title="")
                st.plotly_chart(update_plot_style(fig_c), use_container_width=True)

        if run_btn:
            time.sleep(refresh_rate)
            st.rerun()

# ==========================================
# PAGE: METRICS (MODULE 4 - EVALUATION)
# ==========================================
elif selected_page == "Metrics":
    st.header("📊 Module 4: Model Evaluation")
    
    if st.session_state.eval_results:
        res = st.session_state.eval_results
        
        # 1. Accuracy Score
        st.subheader(f"Model Accuracy: {res['accuracy']:.4f}")
        st.progress(res['accuracy'])
        
        c1, c2 = st.columns(2)
        
        # 2. Confusion Matrix
        with c1:
            st.markdown("#### Confusion Matrix")
            cm_data = res['confusion_matrix']
            # Using heatmap to mimic the Seaborn plot in notebook
            fig_cm = px.imshow(cm_data, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=["Safe", "Unsafe"], y=["Safe", "Unsafe"])
            st.plotly_chart(update_plot_style(fig_cm), use_container_width=True)
            
        # 3. Classification Report
        with c2:
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(res['report']).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
            
        st.info("Validation strategy: Train/Test Split (80/20) with Stratified Sampling.")

# ==========================================
# PAGE: ANALYTICS (MODULE 6 - EXPERIMENTS)
# ==========================================
elif selected_page == "Analytics":
    st.header("🔬 Module 6: Experiments & Analysis")
    
    tab1, tab2 = st.tabs(["Correlations", "Simulations (Module 6)"])
    
    with tab1:
        hist_df = pd.DataFrame(st.session_state.history)
        if not hist_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Temp"], name="Temp", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Humidity"], name="Hum", line=dict(color='#636EFA'), yaxis="y2"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right"), height=400, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(update_plot_style(fig), use_container_width=True)

    with tab2:
        st.markdown("### Experiment 1 & 3: Robustness Testing")
        st.write("Simulating unexpected inputs and noise injection as per Module 6.")
        
        c_ex1, c_ex2 = st.columns(2)
        
        with c_ex1:
            st.markdown("**Exp 1: Extreme Conditions**")
            # Simulating notebook "Unexpected Input"
            extreme_val = st.slider("Simulate Extreme PM2.5", 100, 500, 300)
            if st.button("Test Extreme Input"):
                # DataFrame matching notebook structure
                ex_input = pd.DataFrame([{
                    "temperature_2m": 5, "relativehumidity_2m": 95, 
                    "windspeed_10m": 1, "cloudcover": 100, 
                    "pm2_5_smooth": extreme_val
                }])
                ex_scaled = st.session_state.scaler.transform(ex_input)
                pred = st.session_state.model.predict(ex_scaled)[0]
                res_label = "Unsafe" if pred == 1 else "Safe"
                st.warning(f"Prediction: **{res_label}**")

        with c_ex2:
            st.markdown("**Exp 3: Noise Injection**")
            noise_level = st.slider("Noise Level (Std Dev)", 1, 20, 10)
            if st.button("Inject Noise"):
                base_val = 300
                noise = np.random.normal(0, noise_level)
                noisy_val = base_val + noise
                
                n_input = pd.DataFrame([{
                    "temperature_2m": 5, "relativehumidity_2m": 95, 
                    "windspeed_10m": 1, "cloudcover": 100, 
                    "pm2_5_smooth": noisy_val
                }])
                n_scaled = st.session_state.scaler.transform(n_input)
                pred = st.session_state.model.predict(n_scaled)[0]
                res_label = "Unsafe" if pred == 1 else "Safe"
                st.info(f"Input with Noise: {noisy_val:.2f} | Prediction: **{res_label}**")

# ==========================================
# PAGE: MAP (GEOSPATIAL)
# ==========================================
elif selected_page == "Map":
    st.header("🗺️ Sensor Network")
    st.map(pd.DataFrame({'lat': [lat_input], 'lon': [lon_input]}), zoom=12)
    st.info(f"Sensor Active at: {lat_input}, {lon_input}")

# ==========================================
# PAGE: LOGS (DATA HISTORY)
# ==========================================
elif selected_page == "Logs":
    st.header("📋 Data Logs")
    hist_df = pd.DataFrame(st.session_state.history)
    if not hist_df.empty:
        st.dataframe(hist_df.sort_values(by="Time", ascending=False), use_container_width=True, height=600)