# 🛡️ Sentinel Environment
**Live Environmental Monitoring System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentinel-environment.streamlit.app/)

**🔴 Live Demo:** [Click here to view the Deployed Dashboard](https://sentinel-environment.streamlit.app/)

**Submitted By:**
* **Kanwarajaybir Singh** - 102317223
* **Yati Bhansali** - 102317249

---

## 📌 Project Overview
The **Sentinel Environment** system is a real-time data analysis pipeline designed to monitor environmental conditions and classify air quality safety instantly. By leveraging live sensor data from public APIs, the system processes time-series signals, applies machine learning for classification, and visualizes the results on an interactive dashboard.

---

## 📂 Assessment Module 1: Exploration & Analysis
**Objective:** Explore real-time environmental data modalities and understand sensor behavior.

* **Selected Modality:** Environmental Sensor Time-Series Data (Signal Data).
* **Data Source:** **Open-Meteo** Weather & Air Quality APIs.
* **Parameters Monitored:** Temperature, Humidity, Wind Speed, Cloud Cover, PM2.5.
* **Key Insight:** Environmental data behaves as continuous signals with temporal fluctuations, necessitating preprocessing for stable analysis.

---

## 🧹 Assessment Module 2: Preprocessing
**Objective:** Clean and prepare raw sensor data for accurate modeling.

* **Timestamp Alignment:** Converted and sorted time-series data for chronological consistency.
* **Noise Reduction:** Applied a **Rolling Mean Smoothing (Window=3)** filter to the PM2.5 signal to reduce sensor jitter and short-term fluctuations.
* **Handling Missing Data:** Removed NaN values introduced by the rolling window to ensure dataset integrity.

---

## 🤖 Assessment Module 3: Model Building
**Objective:** Develop a Machine Learning model to classify air quality.

* **Approach:** Machine Learning (Classification).
* **Algorithm:** **Logistic Regression**.
* **Features:** Temperature, Relative Humidity, Wind Speed, Cloud Cover, Smoothed PM2.5.
* **Target Variable:**
    * `0` (Safe): PM2.5 ≤ 100
    * `1` (Unsafe): PM2.5 > 100
* **Outcome:** A trained classification model embedded into the application backend.

---

## 📊 Assessment Module 4: Evaluation
**Objective:** Validate model performance using standard metrics.

* **Evaluation Interface:** Dedicated **"Metrics" Page** in the web app.
* **Key Metrics:** Accuracy, Precision, Recall, F1-Score.
* **Visualization:** Interactive **Confusion Matrix Heatmap**.
* **Validation Strategy:** Stratified Train-Test Split (80/20) to ensure robust performance on unseen data.
* **Result:** The model demonstrated high accuracy and reliable classification capabilities.

---

## ⚡ Assessment Module 5: Deployment
**Objective:** Deploy the model for real-time usage.

* **Platform:** **Streamlit Web Application**.
* **Features:**
    * **Live Dashboard:** Displays real-time metrics with instant updates.
    * **Visualizations:** Speedometer Gauge (Threat Level) and Area Trend Chart.
    * **Control:** User-adjustable "Refresh Rate" (2-60s) and "Live Stream" toggle.
* **Performance:** Low latency (<2s response) with stable predictions.

---

## 🔬 Assessment Module 6: AI Experiments
**Objective:** Test model robustness via controlled simulations.

* **Interface:** **"Analytics" Page** > **Simulations Tab**.
* **Experiment 1 (Unexpected Inputs):** Simulating extreme pollution levels (PM2.5 > 300) correctly triggered "Unsafe" alerts.
* **Experiment 2 (Geospatial Testing):** Changing Latitude/Longitude updated the model context to new global locations successfully.
* **Experiment 3 (Noise Injection):** Adding random Gaussian noise via sliders proved the model's resilience to minor sensor inaccuracies.

---

## 🚀 How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Kanwar019/Sentinel-Environment.git](https://github.com/Kanwar019/Sentinel-Environment.git)
    cd Sentinel-Environment
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch Application:**
    ```bash
    streamlit run app.py
    ```

---
*Developed for the Real-Time Data Analysis Course Assessment.*
