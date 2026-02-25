"""
app.py — Rossmann Sales Forecasting (Simple UI)
Run:  streamlit run app.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Rossmann Forecasting", page_icon="🛒", layout="wide")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = "data/processed/"
MODELS    = "models/"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_val():
    p = PROCESSED + "val_featured.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df["Sales_actual"] = np.expm1(df["Sales_log"])
    return df

@st.cache_data(show_spinner=False)
def load_features():
    p = PROCESSED + "feature_list.csv"
    return pd.read_csv(p)["feature"].tolist() if os.path.exists(p) else []

@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, fname in [("rf", "random_forest.pkl"), ("xgb", "xgboost.pkl")]:
        path = MODELS + fname
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[key] = pickle.load(f)
    return out

val_df   = load_val()
features = load_features()
models   = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛒 Rossmann")
page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🔍 EDA",
    "🤖 Predict",
    "📈 Model Results",
    "📋 Feature Importance",
])
st.sidebar.markdown("---")
st.sidebar.caption("✅ Data loaded" if val_df is not None else "⚠️ No data — run pipeline first")
st.sidebar.caption(f"✅ Models: {', '.join(models.keys()).upper()}" if models else "⚠️ No models found")


# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Rossmann Store Sales Forecasting")
    st.markdown("**Models:** SARIMAX · Random Forest · XGBoost &nbsp;|&nbsp; **Stores:** 1,115 &nbsp;|&nbsp; **Features:** 94")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores",       "1,115")
    c2.metric("Train Rows",   "675,958")
    c3.metric("Features",     "94")
    c4.metric("Best MAE",     "489.65", help="SARIMAX on Store 1")

    st.divider()
    st.subheader("Final Results")
    df_res = pd.DataFrame([
        {"Model": "SARIMAX(2,1,2)+exog", "Scope": "Store 1 only",    "MAE": 489.65,  "RMSE": 653.29},
        {"Model": "XGBoost ⭐",           "Scope": "All 1,115 stores", "MAE": 644.67,  "RMSE": 933.08},
        {"Model": "Random Forest",        "Scope": "All 1,115 stores", "MAE": 736.83,  "RMSE": 1098.15},
    ])
    st.dataframe(
        df_res.style
              .highlight_min(subset=["MAE","RMSE"], color="#bbf7d0")
              .format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}"}),
        hide_index=True, use_container_width=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline Steps")
        for i, s in enumerate([
            "Data preprocessing — merge, clean, filter",
            "Feature engineering — 94 features (lags, rolling, Fourier)",
            "SARIMAX — Store 1 with 16 exogenous features",
            "ML Models — RF + XGBoost on all 1,115 stores",
            "Model comparison — MAE / RMSE / MAPE",
        ], 1):
            st.write(f"**{i}.** {s}")

    with col2:
        st.subheader("Key Findings")
        for f in [
            "Lag features (lag_1, lag_7) are the strongest predictors",
            "Promo is the top categorical driver",
            "SARIMAX beats plain ARIMA by ~30% MAE",
            "XGBoost outperforms RF by ~12% MAE",
            "December & Monday show highest prediction error",
            "SARIMAX beats ML on Store 1 (single-store advantage)",
        ]:
            st.write(f"✅ {f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("EDA Explorer")

    if val_df is None:
        st.error("No data. Run `python src/main.py --step features` first.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    all_stores = sorted(val_df["Store"].unique())
    sel_stores = col1.multiselect("Stores", all_stores, default=all_stores[:4])
    date_range = col2.date_input("Date range",
                     value=(val_df["Date"].min().date(), val_df["Date"].max().date()),
                     min_value=val_df["Date"].min().date(),
                     max_value=val_df["Date"].max().date())
    promo_opt  = col3.selectbox("Promo filter", ["All", "Promo days", "Non-promo days"])

    mask = val_df["Store"].isin(sel_stores)
    if len(date_range) == 2:
        mask &= (val_df["Date"] >= pd.Timestamp(date_range[0])) & \
                (val_df["Date"] <= pd.Timestamp(date_range[1]))
    if promo_opt == "Promo days":      mask &= val_df["Promo"] == 1
    elif promo_opt == "Non-promo days": mask &= val_df["Promo"] == 0
    df = val_df[mask]

    if df.empty:
        st.warning("No data for these filters.")
        st.stop()

    st.caption(f"{len(df):,} rows · {df['Store'].nunique()} store(s)")
    st.divider()

    # Sales over time
    st.subheader("Sales Over Time")
    daily = df.groupby(["Date", "Store"])["Sales_actual"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for s in sel_stores[:8]:
        g = daily[daily["Store"] == s]
        ax.plot(g["Date"], g["Sales_actual"], linewidth=1.2, label=f"Store {s}")
    ax.set_ylabel("Sales"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("By Day of Week")
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        dow = (df.assign(d=df["Date"].dt.dayofweek)
                 .groupby("d")["Sales_actual"].mean()
                 .rename(index=dow_map))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(dow.index, dow.values, color="#3b82f6", edgecolor="white")
        ax.set_ylabel("Avg Sales")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.subheader("Promo vs No Promo")
        pm = df.groupby("Promo")["Sales_actual"].mean()
        if len(pm) == 2:
            lift = (pm[1] - pm[0]) / pm[0] * 100
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(["No Promo", "Promo"], [pm[0], pm[1]],
                   color=["#6b7280","#ef4444"], edgecolor="white", width=0.45)
            ax.set_title(f"Promo lifts sales by {lift:.1f}%")
            ax.set_ylabel("Avg Sales")
            fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Monthly Trend")
    m_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = (df.assign(m=df["Date"].dt.month)
                 .groupby("m")["Sales_actual"].mean()
                 .rename(index=m_map))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(monthly.index, monthly.values, color="#22c55e", edgecolor="white")
    ax.set_ylabel("Avg Sales")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("Raw data table"):
        cols = [c for c in ["Date","Store","Sales_actual","Promo","SchoolHoliday"] if c in df.columns]
        st.dataframe(df[cols].sort_values("Date").head(200), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict":
    st.title("Live Sales Prediction")
    st.caption("Enter store/day features → XGBoost returns a prediction instantly")

    if "xgb" not in models or not features:
        st.error("XGBoost model or feature list not found. Run `python src/main.py --step ml`.")
        st.stop()

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Store & Day**")
        promo    = st.checkbox("Promo active",    value=True)
        is_sch   = st.checkbox("School holiday",  value=False)
        is_pub   = st.checkbox("Public holiday",  value=False)
        dow      = st.selectbox("Day of week", range(1, 8),
                                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])

    with c2:
        st.markdown("**Calendar**")
        month    = st.slider("Month", 1, 12, 6)
        comp_dist= st.number_input("Competition distance (m)", 0, 20000, 1000, 500)

    with c3:
        st.markdown("**Sales History**")
        lag1  = st.number_input("Sales yesterday (lag_1)",  0, 30000, 5800, 100)
        lag7  = st.number_input("Sales 7 days ago (lag_7)", 0, 30000, 5600, 100)
        lag14 = st.number_input("Sales 14 days ago (lag_14)", 0, 30000, 5500, 100)

    st.divider()
    if st.button("🔮  Predict Sales", use_container_width=True, type="primary"):
        row = pd.DataFrame([{c: 0.0 for c in features}])

        field_map = {
            "Promo": int(promo), "SchoolHoliday": int(is_sch),
            "IsPublicHoliday": int(is_pub), "IsMonday": int(dow == 1),
            "IsSaturday": int(dow == 6), "IsQ4": int(month in [10,11,12]),
            "Promo_x_Monday": int(promo and dow == 1),
            "Promo_x_SchoolHol": int(promo and is_sch),
            "DayOfWeek": dow, "Month": month,
            "CompetitionDistance": comp_dist,
            "Sales_lag_1":  np.log1p(lag1),
            "Sales_lag_7":  np.log1p(lag7),
            "Sales_lag_14": np.log1p(lag14),
            "roll_mean_7":  np.log1p((lag1+lag7)/2),
        }
        for d in range(1, 8):
            k = f"DayOfWeek_{d}"
            if k in row.columns: row[k] = int(dow == d)
        for k, v in field_map.items():
            if k in row.columns: row[k] = v

        pred = np.expm1(models["xgb"].predict(row)[0])
        lo, hi = pred * 0.9, pred * 1.1

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Sales",  f"€ {pred:,.0f}")
        r2.metric("Low  (−10%)",      f"€ {lo:,.0f}")
        r3.metric("High (+10%)",      f"€ {hi:,.0f}")
        st.info(f"**Summary:** {'Promo' if promo else 'No promo'} · "
                f"{'School holiday · ' if is_sch else ''}"
                f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow-1]} · "
                f"Month {month} · lag_1 = €{lag1:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.title("Model Results")
    st.divider()

    results = pd.DataFrame([
        {"Model": "SARIMAX+exog", "Scope": "Store 1",      "MAE": 489.65,  "RMSE": 653.29},
        {"Model": "XGBoost",      "Scope": "All 1,115",    "MAE": 644.67,  "RMSE": 933.08},
        {"Model": "Rnd Forest",   "Scope": "All 1,115",    "MAE": 736.83,  "RMSE": 1098.15},
    ])

    c1, c2, c3 = st.columns(3)
    c1.metric("SARIMAX MAE",  "489.65", "Store 1 only",     delta_color="off")
    c2.metric("XGBoost MAE",  "644.67", "All 1,115 stores", delta_color="off")
    c3.metric("RF MAE",       "736.83", "All 1,115 stores", delta_color="off")

    st.divider()
    colors = ["#22c55e", "#ef4444", "#3b82f6"]
    col1, col2 = st.columns(2)

    for col, metric in zip([col1, col2], ["MAE", "RMSE"]):
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(results["Model"], results[metric],
                          color=colors, edgecolor="white", width=0.5)
            for bar, val in zip(bars, results[metric]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 5,
                        f"{val:,.0f}", ha="center", fontweight="bold", fontsize=10)
            ax.set_title(f"{metric} — lower is better")
            ax.set_ylabel(metric)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.dataframe(
        results.style
               .highlight_min(subset=["MAE","RMSE"], color="#bbf7d0")
               .format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}"}),
        hide_index=True, use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Feature Importance":
    st.title("Feature Importance")

    if not models or not features:
        st.error("Models or feature list not found. Run `python src/main.py --step ml` first.")
        st.stop()

    top_n = st.slider("Top N features", 5, 30, 20)
    st.divider()

    col1, col2 = st.columns(2)
    for col, key, color, label in [
        (col1, "rf",  "#3b82f6", "Random Forest"),
        (col2, "xgb", "#ef4444", "XGBoost"),
    ]:
        if key not in models:
            col.warning(f"{label} not loaded.")
            continue
        imp = (pd.Series(models[key].feature_importances_, index=features)
                 .nlargest(top_n)
                 .sort_values())
        with col:
            st.subheader(label)
            fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.32)))
            ax.barh(imp.index, imp.values, color=color, edgecolor="white")
            ax.set_xlabel("Importance")
            fig.tight_layout(); st.pyplot(fig); plt.close()
            with st.expander("Table"):
                st.dataframe(
                    imp.reset_index().rename(columns={"index":"Feature", 0:"Importance"})
                       .style.format({"Importance": "{:.5f}"}),
                    hide_index=True, use_container_width=True,
                )
