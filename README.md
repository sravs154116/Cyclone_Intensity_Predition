# 🌪️ Cyclone Intensity Prediction using Artificial Intelligence  

## 📖 Overview
This project applies **Artificial Intelligence (AI)** techniques to predict cyclone intensity.  
By combining **Machine Learning (ML)** and **Deep Learning (DL)** models, the system is able to:  
- Classify whether a cyclone is likely to occur.  
- Forecast the next wind speed using time-series data.  

The AI-based approach helps in **early disaster warning**, **climate risk reduction**, and **decision-making support** for authorities.  

---

## 📌 Problem Statement
Cyclones cause massive destruction to life and property.  
Predicting their intensity in advance can help authorities take preventive measures and minimize disaster impacts.  

---

## 🎯 Objectives
- To apply **AI techniques** for cyclone prediction.  
- To build a **classification model** that identifies cyclone vs. non-cyclone.  
- To build a **forecasting model** to predict future wind speed.  
- To evaluate AI models for accuracy and reliability in disaster management.  

---

## 📂 Dataset
- **Source**: Atlantic Hurricane Dataset (CSV file)  
- **Key Features**:
  - `Maximum Wind`  
  - `Minimum Pressure`  
  - `Latitude`, `Longitude`, `Date`, `Time`, `Status`  

- **Target Variables**:
  - `Cyclone` (Binary: 1 = Cyclone, 0 = Not Cyclone)  
  - Predicted `Next Wind Speed` (knots)  

---

## 🛠️ Methodology
1. **Data Preprocessing**  
   - Handle missing values  
   - Select important features  
   - Create binary labels  

2. **AI Models Used**  
   - **Machine Learning (Random Forest Classifier)** → cyclone classification  
   - **Deep Learning (LSTM Model)** → wind speed forecasting  

3. **Evaluation**  
   - Classification Accuracy & Reports  
   - Prediction graphs for wind speed  

---

## 📊 Results
- ✅ AI-based classification successfully identified cyclone vs non-cyclone.  
- ✅ LSTM forecasted wind speed trends (example: **41.87 knots predicted**).  
- ✅ Demonstrated the power of AI in disaster management systems.  

---

## 🚀 Future Enhancements
- Integrate **real-time weather data** from APIs.  
- Extend to **multi-hazard prediction** (cyclones, floods, storms).  
- Deploy as an **AI-powered web application** for early warning systems.  

---

## 📦 Installation & Usage
Clone the repository:

```bash
git clone https://github.com/your-username/cyclone-intensity-prediction.git
cd cyclone-intensity-prediction
