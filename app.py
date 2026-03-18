"""
app.py  –  Logistics Optimization System
=========================================
Launch:
    streamlit run app.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from route_optimizer import (
    get_all_route_options, get_risk_label,
    WEATHER_RISK, MODE_SLA, REGION_COORDS,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LogiSense – Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  – dark industrial theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* --- main background --- */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
}

/* metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e6edf3; font-size: 1.6rem; font-weight: 700; }

/* section headers */
h1 { color: #58a6ff !important; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #e6edf3 !important; font-weight: 600; }
h3 { color: #8b949e !important; font-weight: 500; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* divider */
hr { border-color: #30363d; margin: 1.5rem 0; }

/* select / input */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    background: #0d1117 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #8b949e;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.8rem !important;
    transition: all .2s ease;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }

/* dataframes */
.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

/* info / warning boxes */
.stAlert { border-radius: 8px; border: 1px solid #30363d; }

/* mono font for codes */
code { font-family: 'JetBrains Mono', monospace !important; }

/* risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.risk-low      { background: #1a4731; color: #3fb950; border: 1px solid #238636; }
.risk-medium   { background: #3d2e00; color: #d29922; border: 1px solid #9e6a03; }
.risk-high     { background: #4d1f00; color: #f78166; border: 1px solid #b62324; }
.risk-critical { background: #5c0000; color: #ff7b72; border: 1px solid #da3633; }

/* card style */
.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR = "models"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(f"{MODEL_DIR}/delay_model.pkl"):
        return None, None, None, None, None
    with open(f"{MODEL_DIR}/delay_model.pkl",     "rb") as f: model       = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoders.pkl",  "rb") as f: encoders    = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_columns.pkl", "rb") as f: feat_cols   = pickle.load(f)
    with open(f"{MODEL_DIR}/model_metrics.pkl",   "rb") as f: metrics     = pickle.load(f)
    with open(f"{MODEL_DIR}/eda_stats.pkl",       "rb") as f: eda_stats   = pickle.load(f)
    return model, encoders, feat_cols, metrics, eda_stats

@st.cache_data
def load_data():
    df = pd.read_csv("Delivery_Logistics.csv")
    def ns_to_h(c): return pd.to_numeric(c, errors="coerce")
    df["delivery_time_hours"] = ns_to_h(df["delivery_time_hours"])
    df["expected_time_hours"] = ns_to_h(df["expected_time_hours"])
    for c in ["delivery_time_hours", "expected_time_hours"]:
        med = df[c][df[c] > 0].median()
        df[c] = df[c].replace(0, med).fillna(med)
    df["is_delayed"]    = (df["delayed"] == "yes").astype(int)
    df["time_ratio"]    = df["delivery_time_hours"] / (df["expected_time_hours"] + 1e-6)
    df["cost_per_km"]   = df["delivery_cost"] / (df["distance_km"] + 1e-6)
    df["weight_distance"]= df["package_weight_kg"] * df["distance_km"]
    return df

model, encoders, feat_cols, metrics, eda_stats = load_artifacts()
df = load_data()

MODEL_READY = model is not None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚚 LogiSense")
    st.markdown("**Delivery Intelligence Platform**")
    st.markdown("---")

    if MODEL_READY:
        st.success("✅ ML Model Loaded")
        st.metric("Model Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        st.metric("ROC-AUC Score",   f"{metrics['roc_auc']:.3f}")
        st.metric("Training Samples", f"{metrics['train_size']:,}")
    else:
        st.error("⚠️ Model not trained yet")
        st.code("python train_model.py", language="bash")

    st.markdown("---")
    st.markdown("### Dataset Stats")
    st.metric("Total Deliveries", f"{len(df):,}")
    st.metric("Delay Rate",       f"{df['is_delayed'].mean()*100:.1f}%")
    st.metric("Avg Distance",     f"{df['distance_km'].mean():.0f} km")
    st.markdown("---")
    st.caption("Amazon Delivery Logistics Dataset · Kaggle")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🚚 LogiSense")
st.markdown("### Delivery Intelligence & Route Optimization System")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Delay Predictor",
    "🗺️ Route Optimizer",
    "📊 Analytics Dashboard",
    "🤖 Model Insights",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – DELAY PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Delay Risk Predictor")
    st.markdown("Enter delivery parameters to get an instant delay probability assessment.")

    if not MODEL_READY:
        st.warning("Please run `python train_model.py` first to generate the model.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📦 Package Details")
            package_type   = st.selectbox("Package Type",
                ["automobile parts","clothing","cosmetics","documents",
                 "electronics","fragile items","furniture","groceries",
                 "pharmacy"])
            weight         = st.slider("Package Weight (kg)", 0.1, 50.0, 10.0, 0.1)
            delivery_mode  = st.selectbox("Delivery Mode",
                ["same day","express","two day","standard"])

        with col2:
            st.markdown("#### 🚛 Logistics")
            partner        = st.selectbox("Delivery Partner",
                ["amazon logistics","blue dart","delhivery","dhl",
                 "ecom express","ekart","fedex","shadowfax","xpressbees"])
            vehicle        = st.selectbox("Vehicle Type",
                ["bike","ev bike","scooter","van","ev van","truck"])
            region         = st.selectbox("Region",
                ["central","east","north","south","west"])

        with col3:
            st.markdown("#### 🌤️ Environment")
            weather        = st.selectbox("Weather Condition",
                ["clear","cold","foggy","hot","rainy","stormy"])
            distance       = st.slider("Distance (km)", 5.0, 300.0, 100.0, 5.0)
            delivery_cost  = st.number_input("Delivery Cost (₹)", 100.0, 2000.0, 600.0, 10.0)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Delay Risk", use_container_width=True)

        if predict_btn:
            # Encode inputs
            cat_inputs = {
                "delivery_partner":  partner,
                "package_type":      package_type,
                "vehicle_type":      vehicle,
                "delivery_mode":     delivery_mode,
                "region":            region,
                "weather_condition": weather,
            }

            row = {}
            for col_name, val in cat_inputs.items():
                le = encoders[col_name]
                try:
                    row[col_name + "_enc"] = le.transform([val])[0]
                except ValueError:
                    row[col_name + "_enc"] = 0

            # Compute derived features
            time_ratio     = 1.2   # approximate – unknown actual delivery time
            cost_per_km    = delivery_cost / (distance + 1e-6)
            weight_distance= weight * distance

            row["distance_km"]        = distance
            row["package_weight_kg"]  = weight
            row["delivery_cost"]      = delivery_cost
            row["time_ratio"]         = time_ratio
            row["cost_per_km"]        = cost_per_km
            row["weight_distance"]    = weight_distance

            X_input = pd.DataFrame([row])[feat_cols]
            prob    = model.predict_proba(X_input)[0][1]
            label, emoji = get_risk_label(prob)

            # ── result display ──
            res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

            with res_col1:
                st.metric("Delay Probability", f"{prob*100:.1f}%")

            with res_col2:
                risk_class = label.lower().replace(" ", "-")
                st.markdown(
                    f'<br><span class="risk-badge risk-{risk_class}">'
                    f'{emoji} {label}</span>',
                    unsafe_allow_html=True
                )

            with res_col3:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                        "bar":  {"color": "#58a6ff"},
                        "steps": [
                            {"range": [0,  30], "color": "#1a4731"},
                            {"range": [30, 55], "color": "#3d2e00"},
                            {"range": [55, 75], "color": "#4d1f00"},
                            {"range": [75,100], "color": "#5c0000"},
                        ],
                        "threshold": {
                            "line":  {"color": "#ff7b72", "width": 3},
                            "thickness": 0.75,
                            "value": prob * 100,
                        },
                        "bgcolor": "#161b22",
                        "bordercolor": "#30363d",
                    },
                    title={"text": "Risk Gauge", "font": {"color": "#8b949e", "size": 13}},
                ))
                fig.update_layout(
                    paper_bgcolor="#0d1117",
                    font_color="#e6edf3",
                    height=220,
                    margin=dict(t=40, b=0, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── risk factors ──
            st.markdown("#### ⚠️ Risk Factor Breakdown")
            factors = {
                "Weather Risk":    WEATHER_RISK.get(weather, 0),
                "Distance Load":   min(distance / 300, 1.0),
                "Weight Stress":   min(weight / 50, 1.0),
                "Mode Pressure":   {"express": 0.6, "same day": 0.5,
                                    "two day": 0.2, "standard": 0.1}.get(delivery_mode, 0.3),
            }
            fcols = st.columns(len(factors))
            for i, (fname, fval) in enumerate(factors.items()):
                with fcols[i]:
                    st.metric(fname, f"{fval*100:.0f}%")

            # ── recommendation ──
            st.markdown("#### 💡 Recommendation")
            if prob < 0.30:
                st.success("✅ Low delay risk. Proceed with current configuration.")
            elif prob < 0.55:
                st.warning(
                    f"⚠️ Moderate risk. Consider upgrading to a more sheltered vehicle "
                    f"or switching delivery mode. Weather ({weather}) is a contributing factor."
                )
            elif prob < 0.75:
                st.error(
                    f"🟠 High delay risk! Recommend switching to a protected vehicle "
                    f"(van/truck), validating SLA feasibility for {delivery_mode} mode, "
                    f"and alerting the customer proactively."
                )
            else:
                st.error(
                    f"🔴 Critical delay risk! Strongly recommend rescheduling, "
                    f"selecting a different route/region, or postponing until weather improves."
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – ROUTE OPTIMIZER
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Route Optimizer")
    st.markdown("Compare vehicle options and find the optimal route configuration.")

    rc1, rc2 = st.columns(2)

    with rc1:
        r_origin      = st.selectbox("Origin Region",      list(REGION_COORDS.keys()), key="r_orig")
        r_dest        = st.selectbox("Destination Region",
                            [r for r in REGION_COORDS if r != r_origin], key="r_dest")
        r_package     = st.selectbox("Package Type",
                            ["automobile parts","clothing","cosmetics","documents",
                             "electronics","fragile items","furniture","groceries","pharmacy"],
                            key="r_pkg")

    with rc2:
        r_weather     = st.selectbox("Current Weather", list(WEATHER_RISK.keys()), key="r_weather")
        r_mode        = st.selectbox("Delivery Mode",   list(MODE_SLA.keys()),     key="r_mode")
        r_weight      = st.slider("Weight (kg)", 0.1, 50.0, 5.0, 0.1, key="r_wgt")

    # Compute straight-line distance between regions
    from route_optimizer import haversine_km
    lat1, lon1 = REGION_COORDS[r_origin]
    lat2, lon2 = REGION_COORDS[r_dest]
    base_dist  = haversine_km(lat1, lon1, lat2, lon2)

    st.info(f"📍 Estimated inter-region distance: **{base_dist:.0f} km**  "
            f"({r_origin.title()} → {r_dest.title()})")

    optimize_btn = st.button("🗺️ Optimize Routes", use_container_width=True)

    if optimize_btn or True:   # always show on first load
        options = get_all_route_options(
            distance_km=base_dist,
            weather=r_weather,
            delivery_mode=r_mode,
            package_type=r_package,
            weight_kg=r_weight,
        )

        # ── leaderboard table ──
        st.markdown("#### 🏆 Vehicle Ranking")
        rows = []
        for i, opt in enumerate(options):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣"][i]
            rows.append({
                "Rank":            medal,
                "Vehicle":         opt["vehicle"].title(),
                "Est. Time (hr)":  opt["est_time_hr"],
                "Est. Cost (₹)":   f"₹{opt['est_cost_inr']:,.0f}",
                "Weather Risk":    f"{opt['weather_risk']}%",
                "SLA Feasible":    "✅" if opt["sla_feasible"] else "❌",
                "Score":           f"{opt['composite_score']:.3f}",
            })
        route_df = pd.DataFrame(rows)
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        # ── recommended option ──
        best = options[0]
        st.markdown("#### ✅ Optimal Choice")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Best Vehicle",   best["vehicle"].title())
        b2.metric("Estimated Time", f"{best['est_time_hr']} hr")
        b3.metric("Estimated Cost", f"₹{best['est_cost_inr']:,.0f}")
        b4.metric("Composite Score",f"{best['composite_score']:.3f}")

        # ── time vs cost scatter ──
        st.markdown("#### 📈 Time vs Cost Trade-off")
        scatter_df = pd.DataFrame(options)
        scatter_df["vehicle_label"] = scatter_df["vehicle"].str.title()
        scatter_df["SLA"] = scatter_df["sla_feasible"].map({True: "Feasible", False: "Not Feasible"})

        fig_sc = px.scatter(
            scatter_df,
            x="est_time_hr",
            y="est_cost_inr",
            text="vehicle_label",
            color="SLA",
            size="composite_score",
            size_max=30,
            color_discrete_map={"Feasible": "#3fb950", "Not Feasible": "#f78166"},
            labels={"est_time_hr": "Estimated Time (hours)",
                    "est_cost_inr": "Estimated Cost (₹)"},
        )
        fig_sc.update_traces(textposition="top center",
                             textfont=dict(color="#e6edf3", size=11))
        fig_sc.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=380,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── map of regions ──
        st.markdown("#### 🗺️ Route Map (India Regions)")
        map_data = {
            "Region": list(REGION_COORDS.keys()),
            "lat":    [v[0] for v in REGION_COORDS.values()],
            "lon":    [v[1] for v in REGION_COORDS.values()],
            "role":   ["Origin" if r == r_origin else
                       "Destination" if r == r_dest else "Other"
                       for r in REGION_COORDS],
        }
        map_df = pd.DataFrame(map_data)
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon",
            text="Region", color="role",
            color_discrete_map={"Origin":"#58a6ff","Destination":"#3fb950","Other":"#6e7681"},
            size=[20 if r in (r_origin, r_dest) else 12 for r in REGION_COORDS],
            zoom=3.8, center={"lat": 20.5, "lon": 79.0},
            mapbox_style="carto-darkmatter",
            height=380,
        )
        fig_map.update_layout(
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – ANALYTICS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Analytics Dashboard")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Deliveries",  f"{len(df):,}")
    k2.metric("Delayed",           f"{df['is_delayed'].sum():,}")
    k3.metric("Delay Rate",        f"{df['is_delayed'].mean()*100:.1f}%")
    k4.metric("Avg Distance",      f"{df['distance_km'].mean():.0f} km")
    k5.metric("Avg Cost",          f"₹{df['delivery_cost'].mean():.0f}")

    st.markdown("---")

    # Row 1: Delay by weather + delay by vehicle
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        weather_delay = df.groupby("weather_condition")["is_delayed"].mean().reset_index()
        weather_delay.columns = ["Weather", "Delay Rate"]
        weather_delay = weather_delay.sort_values("Delay Rate", ascending=False)
        fig1 = px.bar(
            weather_delay, x="Weather", y="Delay Rate",
            title="Delay Rate by Weather Condition",
            color="Delay Rate",
            color_continuous_scale=[[0,"#238636"],[0.5,"#d29922"],[1,"#da3633"]],
        )
        fig1.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        vehicle_delay = df.groupby("vehicle_type")["is_delayed"].mean().reset_index()
        vehicle_delay.columns = ["Vehicle", "Delay Rate"]
        vehicle_delay = vehicle_delay.sort_values("Delay Rate", ascending=False)
        fig2 = px.bar(
            vehicle_delay, x="Vehicle", y="Delay Rate",
            title="Delay Rate by Vehicle Type",
            color="Delay Rate",
            color_continuous_scale=[[0,"#238636"],[0.5,"#d29922"],[1,"#da3633"]],
        )
        fig2.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            yaxis=dict(tickformat=".0%", gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Delay by mode + delay by region
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        mode_delay = df.groupby("delivery_mode")["is_delayed"].mean().reset_index()
        mode_delay.columns = ["Mode", "Delay Rate"]
        fig3 = px.pie(
            mode_delay, names="Mode", values="Delay Rate",
            title="Delay Share by Delivery Mode",
            hole=0.5,
            color_discrete_sequence=["#58a6ff","#3fb950","#d29922","#f78166"],
        )
        fig3.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3")
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        region_delay = df.groupby("region")["is_delayed"].mean().reset_index()
        region_delay.columns = ["Region", "Delay Rate"]
        fig4 = px.bar(
            region_delay.sort_values("Delay Rate", ascending=True),
            y="Region", x="Delay Rate",
            title="Delay Rate by Region",
            orientation="h",
            color="Delay Rate",
            color_continuous_scale=[[0,"#238636"],[0.5,"#d29922"],[1,"#da3633"]],
        )
        fig4.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            xaxis=dict(tickformat=".0%", gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Distance distribution + partner performance
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig5 = px.histogram(
            df, x="distance_km", nbins=40,
            title="Distance Distribution",
            color_discrete_sequence=["#58a6ff"],
        )
        fig5.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with r3c2:
        partner_data = df.groupby("delivery_partner").agg(
            delay_rate=("is_delayed","mean"),
            avg_rating=("delivery_rating","mean"),
            total=("delivery_id","count"),
        ).reset_index()
        fig6 = px.scatter(
            partner_data, x="avg_rating", y="delay_rate",
            text="delivery_partner", size="total",
            title="Partner: Rating vs Delay Rate",
            color="delay_rate",
            color_continuous_scale=[[0,"#238636"],[1,"#da3633"]],
        )
        fig6.update_traces(textposition="top center",
                           textfont=dict(color="#e6edf3", size=9))
        fig6.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # Heatmap – weather × vehicle
    st.markdown("#### 🔥 Weather × Vehicle Delay Rate Heatmap")
    hm = df.groupby(["weather_condition","vehicle_type"])["is_delayed"].mean().unstack()
    fig7 = px.imshow(
        hm, text_auto=".0%",
        title="Delay Rate: Weather vs Vehicle",
        color_continuous_scale=[[0,"#238636"],[0.5,"#d29922"],[1,"#da3633"]],
        aspect="auto",
    )
    fig7.update_layout(
        paper_bgcolor="#0d1117", font_color="#e6edf3",
        xaxis_title="Vehicle Type", yaxis_title="Weather",
    )
    st.plotly_chart(fig7, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – MODEL INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Model Insights")

    if not MODEL_READY:
        st.warning("Run `python train_model.py` to generate model insights.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",          f"{metrics['accuracy']*100:.2f}%")
        m2.metric("ROC-AUC",           f"{metrics['roc_auc']:.4f}")
        m3.metric("Precision (Delayed)",
                  f"{metrics['report']['1']['precision']*100:.1f}%")
        m4.metric("Recall (Delayed)",
                  f"{metrics['report']['1']['recall']*100:.1f}%")

        st.markdown("---")
        mc1, mc2 = st.columns(2)

        with mc1:
            # Feature importance
            fi = metrics["feature_importances"]
            fi_df = pd.DataFrame({"Feature": list(fi.keys()),
                                  "Importance": list(fi.values())})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
            fig_fi = px.bar(
                fi_df, y="Feature", x="Importance",
                orientation="h",
                title="Top Feature Importances",
                color="Importance",
                color_continuous_scale=[[0,"#21262d"],[1,"#58a6ff"]],
            )
            fig_fi.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font_color="#e6edf3", showlegend=False,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with mc2:
            # Confusion matrix
            cm_arr = np.array(metrics["confusion_matrix"])
            cm_labels = ["Not Delayed", "Delayed"]
            fig_cm = px.imshow(
                cm_arr,
                x=cm_labels, y=cm_labels,
                text_auto=True,
                title="Confusion Matrix",
                color_continuous_scale=[[0,"#161b22"],[1,"#58a6ff"]],
                labels={"x": "Predicted", "y": "Actual"},
            )
            fig_cm.update_layout(
                paper_bgcolor="#0d1117", font_color="#e6edf3",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        st.markdown("#### 📋 Classification Report")
        report_dict = metrics["report"]
        report_rows = []
        for cls_key in ["0", "1", "macro avg", "weighted avg"]:
            if cls_key in report_dict:
                rd = report_dict[cls_key]
                label = {"0":"Not Delayed","1":"Delayed"}.get(cls_key, cls_key.title())
                report_rows.append({
                    "Class":     label,
                    "Precision": f"{rd['precision']:.4f}",
                    "Recall":    f"{rd['recall']:.4f}",
                    "F1-Score":  f"{rd['f1-score']:.4f}",
                    "Support":   int(rd.get("support", 0)),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        # Model description
        st.markdown("#### 🤖 Model Architecture")
        st.markdown("""
<div class="info-card">

**Algorithm:** Random Forest Classifier (200 estimators, max depth 12)  
**Class Balancing:** `class_weight='balanced'` to handle 73/27 imbalance  
**Feature Engineering:**
- Label-encoded categoricals: partner, package type, vehicle, mode, region, weather  
- Numeric: distance, weight, cost, time_ratio, cost_per_km, weight×distance

**Training:** 80/20 stratified split · All 25,000 records used

</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center style='color:#6e7681; font-size:0.8rem;'>"
    "LogiSense · Logistics Optimization System · Amazon Delivery Dataset"
    "</center>",
    unsafe_allow_html=True,
)
