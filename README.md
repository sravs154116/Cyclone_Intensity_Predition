# 🌪️ Cyclone Intensity Prediction using Artificial Intelligence  

## 📖 Overview  
This project applies **Artificial Intelligence (AI)** techniques to predict cyclone intensity.  
It integrates **Machine Learning (ML)** for cyclone classification and **Deep Learning (DL)** for wind speed forecasting, using historical cyclone data.  
The system is deployed as an **interactive Streamlit application**, making it practical for real-world disaster management.  

---

## 📌 Problem Statement  
Cyclones pose a **serious threat to life and property**, and accurate early prediction is crucial for disaster preparedness.  

This project addresses the challenge by:  
- **Classifying cyclone occurrence** (Cyclone vs Non-Cyclone)  
- **Forecasting wind intensity progression** using time-series data  

Approach:  
- **Random Forest Classifier (ML)** → Identifies cyclone conditions based on wind speed and pressure.  
- **LSTM Model (DL)** → Forecasts the **next-step wind speed** to estimate cyclone strength progression.  

The solution combines **classification + forecasting**, making it a **robust AI-driven tool** for cyclone intensity prediction.  

---

## 🎯 Objectives  
- Apply **AI techniques** for cyclone prediction.  
- Build a **Random Forest classifier** for cyclone occurrence detection.  
- Build an **LSTM forecasting model** for wind intensity prediction.  
- Provide an **interactive Streamlit dashboard** for usability.  
- Support **early warnings and disaster planning** for authorities.  

---

## 📂 Dataset  
- **Source**: Atlantic Hurricane Dataset (`atlantic.csv`)  
- **Features**:  
  - `Maximum Wind`, `Minimum Pressure`  
  - `Latitude`, `Longitude`  
  - `Date`, `Time`, `Status`  
- **Target Variables**:  
  - `Cyclone` (Binary: 1 = Cyclone if wind ≥ 64 knots, 0 = Not Cyclone)  
  - `Next Wind Speed` (forecasted value in knots)  

---

## 🛠️ Methodology  

### 🔍 Data Exploration  
- Loaded `atlantic.csv`, inspected features, and checked missing values.  
- Visualized **correlations, distributions, and outliers**.  

### ⚙️ Data Preprocessing  
- Handled missing values.  
- Scaled numeric features (`Maximum Wind`, `Minimum Pressure`).  
- Created binary cyclone labels.  

### 🌪️ Cyclone Classification (Random Forest)  
- Split dataset into train/test sets.  
- Trained a **Random Forest Classifier**.  
- Evaluated accuracy and classification metrics (Precision, Recall, F1-score).  

### 📈 Cyclone Intensity Forecasting (LSTM)  
- Normalized wind speed values.  
- Generated time sequences with **TimeseriesGenerator**.  
- Built an **LSTM model** to predict **next-step wind speed**.  

### 💻 Deployment (Streamlit App)  
- Designed a user interface where users can:  
  - Input **Maximum Wind** and **Minimum Pressure** values.  
  - Get **cyclone classification** (Cyclone / Not Cyclone).  
  - View the **forecasted next wind intensity**.  

---

## 🖥️ Technical Stack  

- **Programming Language**: Python  
- **Libraries/Frameworks**:  
  - Data Handling → `pandas`, `numpy`  
  - Visualization → `matplotlib`, `seaborn`  
  - Machine Learning → `scikit-learn` (RandomForestClassifier)  
  - Deep Learning → `TensorFlow`, `Keras` (LSTM)  
  - Preprocessing → `StandardScaler`  
- **Dataset**: Atlantic Hurricane Dataset (`atlantic.csv`)  
- **Environment**:  
  - Jupyter Notebook (experimentation & training)  
  - Visual Studio Code (development & deployment)  
  - Streamlit (web app interface)  

---

## 📊 Results  

- ✅ **Random Forest** → Reliable cyclone classification with strong accuracy.  
- ✅ **LSTM** → Captured sequential wind speed patterns, forecasting values (e.g., **41.87 knots predicted**).  
- ✅ **Streamlit app** → Provided real-time inputs, predictions, and visualizations.  
- ✅ Demonstrated the **value of AI in disaster management systems**.  

---

## ✅ Conclusion  

This project shows how **AI/ML can enhance disaster management** by predicting cyclone occurrence and intensity.  
- **Random Forest** efficiently classified cyclone vs non-cyclone events.  
- **LSTM** successfully forecasted wind speed trends, capturing sequential dependencies.  
- **Streamlit deployment** made the solution accessible and practical.  

Such a system can assist **authorities and disaster management agencies** in:  
- Issuing **early warnings**  
- Planning **evacuation strategies**  
- Reducing risks to **human life and property**  

---

## 🚀 Future Enhancements  

- Integrate **real-time weather APIs** (NOAA, OpenWeatherMap).  
- Extend to **multi-hazard prediction** (cyclones, floods, storms).  
- Enhance **visualization with geospatial maps** (Folium, Plotly).  
- Deploy on **cloud platforms** (Streamlit Cloud, Hugging Face, Heroku).  

---

## 📦 Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/cyclone-intensity-prediction.git
cd cyclone-intensity-prediction
