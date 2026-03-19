# 🚚 LogiSense — Delivery Intelligence & Route Optimization System

> A Machine Learning-powered logistics platform that predicts delivery delays and recommends optimal delivery routes with actionable insights.

---

## 📌 Overview

**LogiSense** is an end-to-end logistics optimization system designed to solve one of the biggest challenges in modern supply chains — **delivery delays**.

It combines:

- 📊 **Machine Learning (Delay Prediction)**
- 🧭 **Route Optimization Engine**
- 📈 **Interactive Analytics Dashboard**

The system predicts delivery risks **before dispatch**, recommends the **best vehicle & route**, and provides **data-driven insights** for logistics operators.

---

## 🎯 Problem Statement

Delivery delays lead to:

- Increased operational costs
- Customer dissatisfaction
- Failed deliveries & reattempts
- Revenue loss

Traditional systems are reactive — **LogiSense is predictive and proactive.**

---

## 💡 Solution

LogiSense provides **three core capabilities**:

### 🔴 1. Delay Prediction

- Uses **Random Forest Classifier**
- Predicts probability of delay **before dispatch**
- Based on: Weather, Distance, Package type & weight, Vehicle type, Delivery mode

### 🟢 2. Route Optimization Engine

- Physics + business-rule based scoring system
- Suggests **optimal vehicle & route**
- Considers: Weather risk, SLA feasibility, Cost efficiency, Package compatibility

### 🔵 3. Analytics Dashboard

- Built using **Plotly + Streamlit**
- Provides insights on: Delay vs Weather, Delay vs Vehicle, Regional performance, Partner efficiency

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Amazon Delivery Logistics Dataset (Kaggle) |
| **Records** | 25,000 |
| **Features** | 15 columns |

### Key Columns

| Column | Description |
|--------|-------------|
| `vehicle_type` | Type of delivery vehicle |
| `delivery_mode` | Mode of delivery |
| `weather_condition` | Weather at time of delivery |
| `distance_km` | Distance in kilometers |
| `package_weight_kg` | Weight of the package |
| `delivery_cost` | Cost of delivery |
| `delivery_time_hours` | Actual delivery time |
| `expected_time_hours` | Estimated delivery time |
| `delayed` | **Target variable** |

### 📈 Dataset Insights

- 26.7% of deliveries are delayed
- Average distance: ~150 km
- Average cost: ₹900

---

## ⚙️ System Architecture

```
train_model.py      → Model Training
route_optimizer.py  → Route Scoring Engine
app.py              → Streamlit Web App
```

### 🔄 Flow

```
Load Dataset → Preprocess & Feature Engineering → Train ML Model → Save Model (.pkl) → Serve via Streamlit
```

---

## 🧠 Machine Learning Model

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles mixed data types
- Robust to outliers
- No feature scaling required
- Provides feature importance

### ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `class_weight` | balanced |

---

## 🧪 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 88.5% |
| **ROC-AUC** | 0.964 |
| **Recall (Delayed)** | 95.7% |
| **Precision** | 71.1% |

> ✅ High recall ensures we catch almost all real delays before they happen.

---

## 🧩 Feature Engineering

### Derived Features

| Feature | Formula |
|---------|---------|
| `time_ratio` | `actual_time / expected_time` |
| `cost_per_km` | `delivery_cost / distance_km` |
| `weight_distance` | `weight × distance` |

- Label Encoding for categorical variables
- 12 total features used for training

---

## 🧭 Route Optimization Engine

### 🧮 Scoring Formula

```
Score = 0.40 × Weather + 0.35 × SLA + 0.15 × Cost - Penalty
```

### Factors

| Factor | Weight |
|--------|--------|
| 🌧️ Weather Risk | 0.40 |
| ⏱️ SLA Margin | 0.35 |
| 💰 Cost Efficiency | 0.15 |
| ⚠️ Heavy Package Penalty | Deducted |

### 🚗 Vehicles Supported

- 🚲 Bike / EV Bike
- 🛵 Scooter
- 🚐 Van / EV Van
- 🚛 Truck

---

## 🌐 Streamlit Application

### 🖥️ 4 Main Tabs

| Tab | Description |
|-----|-------------|
| **1️⃣ Delay Predictor** | Input delivery details → Outputs delay probability + risk level |
| **2️⃣ Route Optimizer** | Suggests best vehicle, shows cost vs time trade-off |
| **3️⃣ Analytics Dashboard** | Interactive charts, heatmaps & KPI metrics |
| **4️⃣ Model Insights** | Feature importance, confusion matrix, performance metrics |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| ML | scikit-learn |
| Data | pandas, numpy |
| Visualization | Plotly |
| Web App | Streamlit |
| BI | Power BI |
| Storage | Pickle |

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/aarya-prabha/LogiSense.git
cd LogiSense
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model

```bash
python train_model.py
```

### 5️⃣ Run Application

```bash
streamlit run app.py
```

App will open at: `http://localhost:8501`

---

## 📈 Key Insights

- 🚨 **Delivery Mode** is the biggest delay factor
- 🌧️ Stormy weather increases delay probability significantly
- 🚲 Bikes are inefficient for heavy long-distance deliveries
- 🔋 EV vehicles provide notable cost savings
- 🤝 Partner reliability directly impacts delivery success

---

## 🔮 Future Scope

- 🌦️ Real-time Weather API integration
- 📍 Google Maps API for live traffic data
- 🚚 Multi-stop route optimization (TSP)
- 🔁 Automated model retraining pipeline
- 📱 Mobile app for delivery agents
- 🤖 Reinforcement Learning for dynamic routing

---

## 🏆 Hackathon Details

| Detail | Info |
|--------|------|
| 🎯 Event | SECE Hackathon (March 2025) |
| 📊 Dataset Size | 25,000 records |
| 🎯 Model Accuracy | 88.5% |
| ⚡ Prediction Speed | < 10 ms |

---

## 📸 Demo & Screenshots

<img width="1600" height="873" alt="image" src="https://github.com/user-attachments/assets/fe626fcc-e142-41e3-a14c-e2880b32d063" />

---

<img width="1600" height="873" alt="image" src="https://github.com/user-attachments/assets/7f0ef850-dd42-45d4-8dd7-fc29712f0726" />

---

<img width="1600" height="873" alt="image" src="https://github.com/user-attachments/assets/a7b2b00f-6654-4e43-8e46-cdba6153082d" />

---

<img width="1600" height="873" alt="image" src="https://github.com/user-attachments/assets/380e5f69-7cbc-41ff-8791-332b00ef112d" />

---

<img width="1600" height="873" alt="image" src="https://github.com/user-attachments/assets/8278f177-1c8c-4332-bd15-ed73726902a3" />

---
---

## 📊 PowerBI Dashboard

<img width="898" height="505" alt="image" src="https://github.com/user-attachments/assets/b902325e-d18d-44f0-86c7-c008ecc8334f" />

---

<img width="898" height="507" alt="image" src="https://github.com/user-attachments/assets/f3b11731-db48-4481-91ed-e3565f2016b8" />



## 🤝 Contributing

Contributions are welcome!
Feel free to **fork**, open **issues**, or submit **pull requests**.

---

## 📄 License

This project is for educational and research purposes.

---

## 🙌 Acknowledgements

- [Kaggle](https://www.kaggle.com/) — Dataset
- [scikit-learn](https://scikit-learn.org/) — ML Framework
- [Streamlit](https://streamlit.io/) — Web App
- [Plotly](https://plotly.com/) — Visualization

---

## ⭐ Support

If you found this project useful:

- ⭐ **Star** the repository
- 🍴 **Fork** it
- 📢 **Share** with others

---
