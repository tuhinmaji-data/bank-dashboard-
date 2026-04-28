import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ECB | Customer Retention Analytics v3",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F7F8FC; }

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 0.85rem; letter-spacing: 0.04em; font-weight: 600;
}

div[data-testid="metric-container"] {
    background: #FFFFFF; border: 1px solid #E4EAF4;
    border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(26,43,74,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.70rem !important; font-weight: 600 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    color: #6B7FA3 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.75rem !important; color: #1A2B4A !important; font-weight: 700 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

.section-title {
    font-family: 'Playfair Display', serif; font-size: 1.15rem;
    color: #1A2B4A; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.02em;
}
.section-sub { font-size: 0.76rem; color: #8093B4; margin-bottom: 14px; }
.custom-hr { border: none; border-top: 1.5px solid #E4EAF4; margin: 18px 0; }

.header-banner {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F4F8 100%);
    border-radius: 16px; padding: 24px 32px; color: #1A2B4A;
    margin-bottom: 24px; display: flex; align-items: center; gap: 20px;
    box-shadow: 0 4px 24px rgba(26,43,74,0.06); border: 1px solid #E4EAF4;
}
.header-title {
    font-family: 'Playfair Display', serif; font-size: 1.6rem;
    font-weight: 700; letter-spacing: -0.02em; margin: 0 0 4px 0; color: #1A2B4A;
}
.header-sub { font-size: 0.80rem; opacity: 0.85; letter-spacing: 0.04em; color: #6B7FA3; }

/* ── Alert / Recommendation boxes ── */
.alert-warning {
    background: #FFF8EC; border-left: 4px solid #F5A623;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.83rem; color: #7A5A1E; margin-bottom: 12px;
}
.alert-info {
    background: #EFF5FF; border-left: 4px solid #3B82F6;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.83rem; color: #1E40AF; margin-bottom: 12px;
}
.alert-danger {
    background: #FEF2F2; border-left: 4px solid #E8604C;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.83rem; color: #991B1B; margin-bottom: 10px;
}
.alert-success {
    background: #F0FDF4; border-left: 4px solid #52C4A0;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.83rem; color: #166534; margin-bottom: 10px;
}
.rec-header {
    font-family: 'Playfair Display', serif; font-size: 1.05rem;
    font-weight: 700; color: #1A2B4A; margin-bottom: 10px;
    padding-bottom: 6px; border-bottom: 2px solid #E4EAF4;
}
.rec-box {
    background: #FFFFFF; border: 1px solid #E4EAF4;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(26,43,74,0.05);
}
.rec-critical { border-left: 4px solid #E8604C !important; }
.rec-warning  { border-left: 4px solid #F5A623 !important; }
.rec-ok       { border-left: 4px solid #52C4A0 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #EDF0F8; border-radius: 10px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 7px;
    font-size: 0.79rem; font-weight: 600; color: #6B7FA3; letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #1A2B4A !important;
    box-shadow: 0 1px 4px rgba(26,43,74,0.12);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
C = {
    "navy":     "#1A2B4A",
    "blue":     "#2455A4",
    "sky":      "#5B8DD9",
    "ice":      "#A8C4EE",
    "mint":     "#52C4A0",
    "amber":    "#F5A623",
    "coral":    "#E8604C",
    "lavender": "#8B9DC3",
    "bg":       "#F7F8FC",
    "white":    "#FFFFFF",
    "slate":    "#6B7FA3",
}

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")

    def engagement_profile(row):
        active   = row["IsActiveMember"] == 1
        products = row["NumOfProducts"]
        balance  = row["Balance"]
        if active and products >= 2:
            return "Active Engaged"
        elif active and products == 1:
            return "Active Low-Product"
        elif not active and balance > 50_000:
            return "Inactive High-Balance"
        else:
            return "Inactive Disengaged"

    df["EngagementProfile"] = df.apply(engagement_profile, axis=1)

    df["RSI"] = (
        df["IsActiveMember"]  * 30 +
        df["NumOfProducts"]   * 15 +
        df["HasCrCard"]       * 10 +
        (df["Tenure"]      / df["Tenure"].max())      * 25 +
        (df["CreditScore"] / df["CreditScore"].max()) * 20
    ).clip(0, 100).round(1)

    df["BalanceTier"] = pd.cut(
        df["Balance"], bins=[-1, 0, 50_000, 100_000, 150_000, np.inf],
        labels=["Zero", "< 50K", "50–100K", "100–150K", "150K+"],
    )
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    )
    df["SalaryTier"] = pd.cut(
        df["EstimatedSalary"], bins=[0, 50_000, 100_000, 150_000, np.inf],
        labels=["< 50K", "50–100K", "100–150K", "150K+"],
    )
    df["AtRiskPremium"] = (
        (df["Balance"] > 100_000) & (df["IsActiveMember"] == 0)
    ).astype(int)

    # ── NEW: Churn Driver Label ──────────────────
    def churn_driver(row):
        if row["IsActiveMember"] == 0:
            return "Inactivity"
        elif row["NumOfProducts"] >= 3:
            return "Product Overload"
        elif row["NumOfProducts"] == 1:
            return "Low Product Depth"
        elif row["Age"] > 50:
            return "Age Risk"
        elif row["Balance"] == 0:
            return "Zero Balance"
        else:
            return "Other"
    df["ChurnDriver"] = df.apply(churn_driver, axis=1)

    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 ECB Analytics")
    st.markdown("**Retention Strategy Platform v3**")
    st.markdown("<hr>", unsafe_allow_html=True)

    geo_opts    = ["All"] + sorted(df["Geography"].unique().tolist())
    selected_geo = st.selectbox("🌍 Geography", geo_opts, index=0)

    gen_opts    = ["All", "Male", "Female"]
    selected_gen = st.selectbox("👥 Gender", gen_opts, index=0)

    activity_map = {"All": None, "Active Only": 1, "Inactive Only": 0}
    selected_act = st.selectbox("🔘 Member Status", list(activity_map.keys()), index=0)

    prod_range = st.slider("📦 Products Held", 1, int(df["NumOfProducts"].max()),
                           (1, int(df["NumOfProducts"].max())))

    bal_range = st.slider("💰 Account Balance (€)", 0, int(df["Balance"].max()),
                          (0, int(df["Balance"].max())), step=5_000, format="€%d")

    ep_opts     = ["All"] + sorted(df["EngagementProfile"].unique().tolist())
    selected_ep = st.selectbox("📊 Engagement Profile", ep_opts, index=0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("European Central Bank  •  Retention Analytics v3.0")

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
fdf = df.copy()
if selected_geo != "All":
    fdf = fdf[fdf["Geography"] == selected_geo]
if selected_gen != "All":
    fdf = fdf[fdf["Gender"] == selected_gen]
if activity_map[selected_act] is not None:
    fdf = fdf[fdf["IsActiveMember"] == activity_map[selected_act]]
fdf = fdf[
    (fdf["NumOfProducts"] >= prod_range[0]) &
    (fdf["NumOfProducts"] <= prod_range[1]) &
    (fdf["Balance"]       >= bal_range[0])  &
    (fdf["Balance"]       <= bal_range[1])
]
if selected_ep != "All":
    fdf = fdf[fdf["EngagementProfile"] == selected_ep]

# ─────────────────────────────────────────────
# CHART THEME HELPER
# ─────────────────────────────────────────────
def bold_theme(f):
    f.update_layout(font=dict(color="#1A2B4A"))
    f.update_xaxes(tickfont=dict(color="#1A2B4A", size=12), title_font=dict(color="#1A2B4A", size=13))
    f.update_yaxes(tickfont=dict(color="#1A2B4A", size=12), title_font=dict(color="#1A2B4A", size=13))
    return f

# ─────────────────────────────────────────────
# ══ FIX 3 — SMART ALERT ENGINE ══════════════
# ─────────────────────────────────────────────
def generate_alerts(fdf, df):
    alerts = []
    churn_rate     = fdf["Exited"].mean() * 100 if len(fdf) > 0 else 0
    active_churn   = fdf[fdf["IsActiveMember"]==1]["Exited"].mean()*100 if len(fdf[fdf["IsActiveMember"]==1])>0 else 0
    inactive_churn = fdf[fdf["IsActiveMember"]==0]["Exited"].mean()*100 if len(fdf[fdf["IsActiveMember"]==0])>0 else 0
    single_pct     = (fdf["NumOfProducts"]==1).mean()*100
    multi_high_churn = fdf[fdf["NumOfProducts"]>=3]["Exited"].mean()*100 if len(fdf[fdf["NumOfProducts"]>=3])>0 else 0
    at_risk_n      = fdf["AtRiskPremium"].sum()
    rsi_q1_pct     = (fdf["RSI"] < 65).mean()*100 if len(fdf)>0 else 0

    # Germany churn check
    if "Germany" in fdf["Geography"].values:
        ger_churn = fdf[fdf["Geography"]=="Germany"]["Exited"].mean()*100
        if ger_churn > 30:
            alerts.append(("critical", "🚨 Germany Churn Crisis",
                f"Germany churn rate is <b>{ger_churn:.1f}%</b> — significantly above the portfolio average. "
                f"<br>→ <b>Action:</b> Deploy dedicated Germany retention workstream immediately. Engage local relationship managers for inactive high-balance segment."))

    # Overall churn
    if churn_rate > 25:
        alerts.append(("critical", "🚨 Churn Rate Critical",
            f"Overall churn rate of <b>{churn_rate:.1f}%</b> exceeds the 25% critical threshold. "
            f"<br>→ <b>Action:</b> Escalate to retention leadership. Run RSI Q1 outreach campaign within 7 days."))
    elif churn_rate > 18:
        alerts.append(("warning", "⚠️ Churn Rate Above Target",
            f"Churn rate <b>{churn_rate:.1f}%</b> exceeds the 18% strategic benchmark. "
            f"<br>→ <b>Action:</b> Review inactive member engagement programmes. Prioritise Q1 RSI customers."))
    else:
        alerts.append(("ok", "✅ Churn Rate On Target",
            f"Churn rate of <b>{churn_rate:.1f}%</b> is within the acceptable threshold. Continue monitoring weekly."))

    # Inactivity gap
    if inactive_churn > 0 and (inactive_churn - active_churn) > 10:
        alerts.append(("critical", "🚨 Inactivity Driving Churn",
            f"Inactive members churn at <b>{inactive_churn:.1f}%</b> vs <b>{active_churn:.1f}%</b> for active — a <b>{inactive_churn-active_churn:.1f}pp gap</b>. "
            f"<br>→ <b>Action:</b> Launch re-engagement campaign for all inactive members. Offer credit card or second product to inactive single-product holders."))

    # Single product concentration
    if single_pct > 50:
        alerts.append(("warning", "⚠️ Single-Product Concentration Risk",
            f"<b>{single_pct:.1f}%</b> of the filtered base holds only 1 product — high churn vulnerability. "
            f"<br>→ <b>Action:</b> Launch 1-to-2 product cross-sell campaign. Target active single-product holders first — 71% churn reduction opportunity."))

    # Product overload mis-selling signal
    if multi_high_churn > 50:
        alerts.append(("critical", "🚨 Product Overload — Mis-Selling Signal",
            f"Customers with 3+ products show <b>{multi_high_churn:.1f}%</b> churn. This is a mis-selling red flag. "
            f"<br>→ <b>Action:</b> Immediately review cross-sell practices. Cap product recommendations at 2 for standard retail customers. Audit 3–4 product accounts."))

    # At-risk premium
    if at_risk_n > 50:
        alerts.append(("critical", "🚨 At-Risk Premium Customers Detected",
            f"<b>{at_risk_n:,}</b> high-balance inactive customers identified — high silent churn risk. "
            f"<br>→ <b>Action:</b> Assign relationship managers to top 30 by balance. Personal outreach within 30 days. Do not use generic email campaigns."))
    elif at_risk_n > 10:
        alerts.append(("warning", "⚠️ Premium Segment Monitoring",
            f"<b>{at_risk_n:,}</b> high-balance inactive customers flagged. Monitor closely. "
            f"<br>→ <b>Action:</b> Schedule quarterly check-ins. Offer advisory service review."))

    # RSI Q1 concentration
    if rsi_q1_pct > 30:
        alerts.append(("warning", "⚠️ High Weak-Relationship Concentration",
            f"<b>{rsi_q1_pct:.1f}%</b> of customers fall in the weakest RSI quartile (score < 65). "
            f"<br>→ <b>Action:</b> Run RSI Q1 engagement blitz. Focus on activity reactivation and second product offer."))

    return alerts

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div style="font-size:2.6rem;">🏦</div>
  <div>
    <div class="header-title">Customer Engagement & Retention Analytics</div>
    <div class="header-sub">EUROPEAN CENTRAL BANK  •  BEHAVIORAL INTELLIGENCE PLATFORM  •  LIVE DASHBOARD  •  v3.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW — 2 rows of 3
# ─────────────────────────────────────────────
total        = len(fdf)
churn_rate   = fdf["Exited"].mean() * 100
active_rate  = fdf["IsActiveMember"].mean() * 100
avg_prod     = fdf["NumOfProducts"].mean()
avg_rsi      = fdf["RSI"].mean()
at_risk_n    = fdf["AtRiskPremium"].sum()
inactive_churn = fdf[fdf["IsActiveMember"]==0]["Exited"].mean()*100 if len(fdf[fdf["IsActiveMember"]==0])>0 else 0
single_churn   = fdf[fdf["NumOfProducts"]==1]["Exited"].mean()*100 if len(fdf[fdf["NumOfProducts"]==1])>0 else 0

row1 = st.columns(3)
row1[0].metric("Total Customers",           f"{total:,}")
row1[1].metric("Overall Churn Rate",         f"{churn_rate:.1f}%",
               delta=f"{churn_rate - df['Exited'].mean()*100:+.1f}pp vs. full", delta_color="inverse")
row1[2].metric("Active Member Rate",         f"{active_rate:.1f}%")

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
row2 = st.columns(3)
row2[0].metric("Avg. Products Held",         f"{avg_prod:.2f}")
row2[1].metric("Avg. Relationship Strength", f"{avg_rsi:.1f}/100")
row2[2].metric("At-Risk Premium Customers",  f"{at_risk_n:,}",
               delta="High Balance + Inactive", delta_color="off")

st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ══ SMART ALERT PANEL ════════════════════════
# ─────────────────────────────────────────────
with st.expander("🔔 Smart Alert & Recommendation Engine — Click to View Live Alerts", expanded=True):
    alerts = generate_alerts(fdf, df)
    if not alerts:
        st.markdown('<div class="alert-success">✅ No critical alerts detected for the current filter selection.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(2)
        for i, (level, title, body) in enumerate(alerts):
            css = {"critical": "alert-danger", "warning": "alert-warning", "ok": "alert-success"}[level]
            cols[i % 2].markdown(
                f'<div class="{css}" style="border-radius:10px; padding:14px 16px; margin-bottom:10px;">'
                f'<b>{title}</b><br><span style="font-size:0.82rem">{body}</span></div>',
                unsafe_allow_html=True
            )

st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS — added new Churn Root Cause tab
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Engagement Overview",
    "📦 Product Utilization",
    "💎 Premium Risk Radar",
    "🧮 Retention Strength",
    "🔍 Churn Root Cause",      # ── NEW ──
    "📋 Customer Explorer",
])

# ══════════════════════════════════════════════
# TAB 1 — ENGAGEMENT OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<p class="section-title">Churn Rate by Engagement Profile</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Which behavioral segments are most at risk?</p>', unsafe_allow_html=True)
        ep_churn = (
            fdf.groupby("EngagementProfile")["Exited"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "Customers"})
            .sort_values("ChurnRate", ascending=True)
        )
        ep_churn["ChurnRate"] *= 100
        fig = px.bar(
            ep_churn, x="ChurnRate", y="EngagementProfile", orientation="h",
            color="ChurnRate",
            color_continuous_scale=[[0, C["ice"]], [0.5, C["sky"]], [1, C["coral"]]],
            text=ep_churn["ChurnRate"].map(lambda v: f"{v:.1f}%"),
            custom_data=["Customers"],
        )
        fig.update_traces(textposition="outside",
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1f}%<br>Customers: %{customdata[0]:,}<extra></extra>")
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis_title="Churn Rate (%)", yaxis_title="",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=180, r=50, t=10, b=30),   # ← FIX: l=180 prevents label clipping
            height=280, font=dict(family="DM Sans", size=12),
            xaxis=dict(gridcolor="#E4EAF4"),
        )
        st.plotly_chart(bold_theme(fig), use_container_width=True, theme=None)

    with col_r:
        st.markdown('<p class="section-title">Engagement Profile Distribution</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Composition of the customer base</p>', unsafe_allow_html=True)
        ep_dist = fdf["EngagementProfile"].value_counts().reset_index()
        ep_dist.columns = ["Profile", "Count"]
        colors = [C["navy"], C["sky"], C["mint"], C["amber"]]
        fig2 = px.pie(ep_dist, values="Count", names="Profile",
                      color_discrete_sequence=colors, hole=0.55)
        fig2.update_traces(textposition="outside", textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>")
        fig2.update_layout(
            showlegend=False, plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=20, r=20, t=10, b=10), height=280,
            font=dict(family="DM Sans", size=11),
        )
        st.plotly_chart(bold_theme(fig2), use_container_width=True, theme=None)

    st.markdown('<p class="section-title">Churn Heatmap: Geography × Activity Status</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Where are inactive customers churning most?</p>', unsafe_allow_html=True)
    geo_act = fdf.groupby(["Geography", "IsActiveMember"])["Exited"].mean().mul(100).round(1).reset_index()
    geo_act["Activity"] = geo_act["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    heat_pivot = geo_act.pivot(index="Geography", columns="Activity", values="Exited")
    fig3 = px.imshow(heat_pivot,
        color_continuous_scale=[[0, "#EFF5FF"], [0.5, C["sky"]], [1, C["coral"]]],
        text_auto=".1f", aspect="auto")
    fig3.update_layout(
        coloraxis_colorbar_title="Churn %",
        plot_bgcolor=C["white"], paper_bgcolor=C["white"],
        margin=dict(l=80, r=10, t=10, b=10), height=200,
        font=dict(family="DM Sans", size=13), xaxis_title="", yaxis_title="",
    )
    st.plotly_chart(bold_theme(fig3), use_container_width=True, theme=None)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="section-title">Churn Rate by Age Group</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Age-driven attrition patterns</p>', unsafe_allow_html=True)
        age_churn = fdf.groupby("AgeGroup", observed=True)["Exited"].agg(["mean","count"]).reset_index()
        age_churn["ChurnRate"] = age_churn["mean"] * 100
        fig4 = px.bar(age_churn, x="AgeGroup", y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale=[[0, C["ice"]], [1, C["navy"]]],
            text=age_churn["ChurnRate"].map(lambda v: f"{v:.1f}%"))
        fig4.update_traces(textposition="outside")
        fig4.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis_title="Age Group", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=30, b=30), height=280,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
        )
        # ── FIX: annotation for highest-risk age group ──
        if len(age_churn) > 0:
            worst = age_churn.loc[age_churn["ChurnRate"].idxmax()]
            fig4.add_annotation(
                x=worst["AgeGroup"], y=worst["ChurnRate"] + 3,
                text=f"⚠ Highest Risk", showarrow=False,
                font=dict(size=11, color=C["coral"], family="DM Sans"),
            )
        st.plotly_chart(bold_theme(fig4), use_container_width=True, theme=None)

    with c2:
        st.markdown('<p class="section-title">Churn Rate by Tenure (Years)</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Does loyalty duration reduce churn?</p>', unsafe_allow_html=True)
        tenure_churn = fdf.groupby("Tenure")["Exited"].mean().mul(100).round(1).reset_index()
        fig5 = px.line(tenure_churn, x="Tenure", y="Exited",
            markers=True, line_shape="spline",
            color_discrete_sequence=[C["blue"]])
        fig5.update_traces(
            line=dict(width=2.5),
            marker=dict(size=7, color=C["sky"], line=dict(width=2, color=C["navy"])))
        fig5.add_hline(y=churn_rate, line_dash="dot", line_color=C["coral"],
                       annotation_text=f"Avg {churn_rate:.1f}%", annotation_position="bottom right")
        fig5.update_layout(
            xaxis_title="Tenure (Years)", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=280,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
        )
        st.plotly_chart(bold_theme(fig5), use_container_width=True, theme=None)

# ══════════════════════════════════════════════
# TAB 2 — PRODUCT UTILIZATION
# ══════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="section-title">Churn Rate by Number of Products</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Does product depth build loyalty?</p>', unsafe_allow_html=True)
        prod_churn = fdf.groupby("NumOfProducts")["Exited"].agg(["mean","count"]).reset_index()
        prod_churn["ChurnRate"] = prod_churn["mean"] * 100
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=prod_churn["NumOfProducts"].astype(str), y=prod_churn["count"],
            name="Customers", marker_color=C["ice"], yaxis="y2", opacity=0.6))
        fig6.add_trace(go.Scatter(
            x=prod_churn["NumOfProducts"].astype(str), y=prod_churn["ChurnRate"],
            name="Churn Rate", mode="lines+markers+text",
            line=dict(color=C["coral"], width=3),
            marker=dict(size=10, color=C["coral"]),
            text=prod_churn["ChurnRate"].map(lambda v: f"{v:.1f}%"),
            textposition="top center"))
        fig6.update_layout(
            yaxis=dict(title="Churn Rate (%)", gridcolor="#E4EAF4"),
            yaxis2=dict(title="# Customers", overlaying="y", side="right", showgrid=False),
            xaxis=dict(title="Number of Products", type="category"),
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=300,
            font=dict(family="DM Sans", size=12), legend=dict(orientation="h", y=1.1),
        )
        # ── FIX: annotation for product sweet spot ──
        fig6.add_annotation(x="2", y=prod_churn[prod_churn["NumOfProducts"]==2]["ChurnRate"].values[0] if 2 in prod_churn["NumOfProducts"].values else 8,
            text="Sweet Spot", showarrow=True, arrowhead=2,
            font=dict(size=11, color=C["mint"]), arrowcolor=C["mint"], ay=-30)
        st.plotly_chart(bold_theme(fig6), use_container_width=True, theme=None)

    with c2:
        st.markdown('<p class="section-title">Credit Card Ownership vs Churn</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Credit Card Stickiness Score</p>', unsafe_allow_html=True)
        cc_churn = fdf.groupby(["HasCrCard","IsActiveMember"])["Exited"].mean().mul(100).round(1).reset_index()
        cc_churn["HasCard"]  = cc_churn["HasCrCard"].map({1:"Has Credit Card", 0:"No Credit Card"})
        cc_churn["Activity"] = cc_churn["IsActiveMember"].map({1:"Active", 0:"Inactive"})
        fig7 = px.bar(cc_churn, x="HasCard", y="Exited", color="Activity",
            barmode="group",
            color_discrete_map={"Active": C["mint"], "Inactive": C["coral"]},
            text=cc_churn["Exited"].map(lambda v: f"{v:.1f}%"))
        fig7.update_traces(textposition="outside")
        fig7.update_layout(
            xaxis_title="", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=300,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
            legend=dict(title="", orientation="h", y=1.1),
        )
        st.plotly_chart(bold_theme(fig7), use_container_width=True, theme=None)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<p class="section-title">Single vs Multi-Product Retention</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Retention differential across geographies</p>', unsafe_allow_html=True)
        fdf2 = fdf.copy()
        fdf2["ProductGroup"] = fdf2["NumOfProducts"].apply(lambda x: "Single Product" if x==1 else "Multi-Product")
        sp_geo = fdf2.groupby(["Geography","ProductGroup"])["Exited"].mean().mul(100).round(1).reset_index()
        fig8 = px.bar(sp_geo, x="Geography", y="Exited", color="ProductGroup", barmode="group",
            color_discrete_map={"Single Product": C["amber"], "Multi-Product": C["mint"]},
            text=sp_geo["Exited"].map(lambda v: f"{v:.1f}%"))
        fig8.update_traces(textposition="outside")
        fig8.update_layout(
            xaxis_title="", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=280,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
            legend=dict(title="", orientation="h", y=1.1),
        )
        st.plotly_chart(bold_theme(fig8), use_container_width=True, theme=None)

    with c4:
        st.markdown('<p class="section-title">Product Depth Index by Churn Status</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Distribution of products for churned vs retained</p>', unsafe_allow_html=True)
        pd_dist = fdf.groupby(["NumOfProducts","Exited"]).size().reset_index(name="Count")
        pd_dist["Status"] = pd_dist["Exited"].map({0:"Retained", 1:"Churned"})
        fig9 = px.bar(pd_dist, x="NumOfProducts", y="Count", color="Status", barmode="stack",
            color_discrete_map={"Retained": C["sky"], "Churned": C["coral"]})
        fig9.update_layout(
            xaxis_title="Products Held", yaxis_title="Customers",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=280,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
            legend=dict(title="", orientation="h", y=1.1),
        )
        st.plotly_chart(bold_theme(fig9), use_container_width=True, theme=None)

# ══════════════════════════════════════════════
# TAB 3 — PREMIUM RISK RADAR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="alert-warning">
    ⚠️ <b>At-Risk Premium Segment:</b> Customers with Balance > €100,000 who are <b>inactive members</b>
    represent a high-value silent churn risk. Immediate engagement intervention is recommended.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="section-title">Balance Tier vs Churn Rate</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Does wealth protect against churn?</p>', unsafe_allow_html=True)
        bt_churn = fdf.groupby("BalanceTier", observed=True)["Exited"].agg(["mean","count"]).reset_index()
        bt_churn["ChurnRate"] = bt_churn["mean"] * 100
        fig10 = px.bar(bt_churn, x="BalanceTier", y="ChurnRate",
            color="ChurnRate",
            color_continuous_scale=[[0, C["mint"]], [0.5, C["amber"]], [1, C["coral"]]],
            text=bt_churn["ChurnRate"].map(lambda v: f"{v:.1f}%"))
        fig10.update_traces(textposition="outside")
        fig10.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Balance Tier (€)", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=280,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
        )
        st.plotly_chart(bold_theme(fig10), use_container_width=True, theme=None)

    with c2:
        st.markdown('<p class="section-title">High-Balance Disengagement Rate</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Inactive members with balance > €100K by region</p>', unsafe_allow_html=True)
        hb = fdf[fdf["Balance"] > 100_000]
        if len(hb) > 0:
            hb_geo = hb.groupby("Geography").agg(
                Total=("CustomerId","count"),
                Inactive=("IsActiveMember", lambda x: (x==0).sum()),
                Churned=("Exited","sum")).reset_index()
            hb_geo["InactiveRate"] = (hb_geo["Inactive"] / hb_geo["Total"] * 100).round(1)
            hb_geo["ChurnRate"]    = (hb_geo["Churned"]  / hb_geo["Total"] * 100).round(1)
            fig11 = go.Figure()
            fig11.add_trace(go.Bar(x=hb_geo["Geography"], y=hb_geo["InactiveRate"],
                name="Inactive Rate", marker_color=C["amber"],
                text=hb_geo["InactiveRate"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
            fig11.add_trace(go.Bar(x=hb_geo["Geography"], y=hb_geo["ChurnRate"],
                name="Churn Rate", marker_color=C["coral"],
                text=hb_geo["ChurnRate"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
            fig11.update_layout(
                barmode="group", xaxis_title="", yaxis_title="Rate (%)",
                plot_bgcolor=C["white"], paper_bgcolor=C["white"],
                margin=dict(l=10, r=10, t=10, b=30), height=280,
                font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(bold_theme(fig11), use_container_width=True, theme=None)
        else:
            st.info("No high-balance customers in current filter selection.")

    # ── FIX: Scatter — reduced to 400 sample, discrete colors, quadrant lines ──
    st.markdown('<p class="section-title">Salary–Balance Matrix</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Quadrant view: identify premium churn clusters and financial commitment mismatches</p>', unsafe_allow_html=True)
    sample = fdf.sample(min(400, len(fdf)), random_state=42)
    sample["Churn Status"] = sample["Exited"].map({0: "Retained", 1: "Churned"})
    fig12 = px.scatter(
        sample, x="EstimatedSalary", y="Balance",
        color="Churn Status",
        color_discrete_map={"Retained": C["sky"], "Churned": C["coral"]},
        opacity=0.70, size_max=8,
        hover_data=["Geography", "NumOfProducts", "Age", "RSI"],
        labels={"EstimatedSalary": "Estimated Salary (€)", "Balance": "Account Balance (€)"},
    )
    # Add quadrant lines
    med_sal = fdf["EstimatedSalary"].median()
    med_bal = fdf["Balance"].median()
    fig12.add_vline(x=med_sal, line_dash="dash", line_color=C["slate"], line_width=1,
                    annotation_text="Median Salary", annotation_position="top right",
                    annotation_font=dict(size=10, color=C["slate"]))
    fig12.add_hline(y=med_bal, line_dash="dash", line_color=C["slate"], line_width=1,
                    annotation_text="Median Balance", annotation_position="top right",
                    annotation_font=dict(size=10, color=C["slate"]))
    fig12.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor=C["white"],
        margin=dict(l=10, r=10, t=10, b=30), height=360,
        font=dict(family="DM Sans", size=12),
        xaxis=dict(gridcolor="#E4EAF4"), yaxis=dict(gridcolor="#E4EAF4"),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(bold_theme(fig12), use_container_width=True, theme=None)

    st.markdown('<p class="section-title">At-Risk Premium Customer List</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">High-balance inactive members — priority outreach candidates</p>', unsafe_allow_html=True)
    atrisk = fdf[fdf["AtRiskPremium"]==1][
        ["CustomerId","Surname","Geography","Gender","Age","Balance",
         "NumOfProducts","Tenure","RSI","Exited"]
    ].sort_values("Balance", ascending=False).head(30)
    atrisk["Balance"] = atrisk["Balance"].map("€{:,.0f}".format)
    atrisk["RSI"]     = atrisk["RSI"].map("{:.1f}".format)
    atrisk["Exited"]  = atrisk["Exited"].map({0: "✅ Retained", 1: "🚨 Churned"})
    st.dataframe(atrisk.reset_index(drop=True), use_container_width=True, height=320)

# ══════════════════════════════════════════════
# TAB 4 — RETENTION STRENGTH
# ══════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="section-title">Relationship Strength Index Distribution</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">RSI = Activity + Products + Card + Tenure + Credit Score</p>', unsafe_allow_html=True)
        fig13 = px.histogram(fdf, x="RSI", color="Exited",
            color_discrete_map={0: C["sky"], 1: C["coral"]},
            barmode="overlay", nbins=30, opacity=0.75,
            labels={"Exited": "Churned", "RSI": "Relationship Strength Index"})
        fig13.update_layout(
            xaxis_title="RSI Score", yaxis_title="Number of Customers",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=300,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
            legend=dict(title="Churned", orientation="h", y=1.1),
        )
        st.plotly_chart(bold_theme(fig13), use_container_width=True, theme=None)

    with c2:
        st.markdown('<p class="section-title">Churn Rate by RSI Quartile</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Higher RSI = lower churn — validating the index</p>', unsafe_allow_html=True)
        fdf3 = fdf.copy()
        if len(fdf3) >= 4:
            try:
                fdf3["RSI_Quartile"] = pd.qcut(fdf3["RSI"], 4, labels=["Q1 Weak","Q2 Fair","Q3 Good","Q4 Strong"])
            except ValueError:
                try:
                    fdf3["RSI_Quartile"] = pd.qcut(fdf3["RSI"].rank(method="first"), 4, labels=["Q1 Weak","Q2 Fair","Q3 Good","Q4 Strong"])
                except:
                    fdf3["RSI_Quartile"] = "Q1 Weak"
        else:
            fdf3["RSI_Quartile"] = "Q1 Weak"
        rsi_q = fdf3.groupby("RSI_Quartile", observed=True)["Exited"].mean().mul(100).round(1).reset_index()
        fig14 = px.bar(rsi_q, x="RSI_Quartile", y="Exited",
            color="Exited",
            color_continuous_scale=[[0, C["mint"]], [1, C["coral"]]],
            text=rsi_q["Exited"].map(lambda v: f"{v:.1f}%"))
        fig14.update_traces(textposition="outside")
        fig14.update_layout(
            coloraxis_showscale=False,
            xaxis_title="RSI Quartile", yaxis_title="Churn Rate (%)",
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=30), height=300,
            font=dict(family="DM Sans", size=12), yaxis=dict(gridcolor="#E4EAF4"),
        )
        st.plotly_chart(bold_theme(fig14), use_container_width=True, theme=None)

    # ── FIX: Radar expanded to full width ──
    st.markdown('<p class="section-title">Sticky Customer Profiling — Radar Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Retained vs churned customers across key engagement dimensions</p>', unsafe_allow_html=True)
    dims       = ["IsActiveMember","NumOfProducts","HasCrCard","Tenure","CreditScore"]
    dim_labels = ["Active Member","Products Held","Has Credit Card","Tenure (yrs)","Credit Score"]
    retained   = fdf[fdf["Exited"]==0][dims].mean()
    churned    = fdf[fdf["Exited"]==1][dims].mean()
    max_vals   = fdf[dims].max()
    retained_n = (retained / max_vals).tolist()
    churned_n  = (churned  / max_vals).tolist()

    ra, rb = st.columns([3, 1])
    with ra:
        fig15 = go.Figure()
        for vals, name, color in [
            (retained_n + [retained_n[0]], "Retained", C["sky"]),
            (churned_n  + [churned_n[0]],  "Churned",  C["coral"]),
        ]:
            fig15.add_trace(go.Scatterpolar(
                r=vals, theta=dim_labels + [dim_labels[0]],
                fill="toself", name=name,
                line=dict(color=color, width=2.5),
                fillcolor=color, opacity=0.22))
        fig15.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#E4EAF4", tickfont=dict(size=10)),
                angularaxis=dict(gridcolor="#E4EAF4", tickfont=dict(size=12)),
                bgcolor=C["white"]),
            showlegend=True, plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=60, r=60, t=40, b=40), height=420,
            font=dict(family="DM Sans", size=12),
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(bold_theme(fig15), use_container_width=True, theme=None)

    with rb:
        st.markdown("**Key Gaps**")
        for dim, dl in zip(dims, dim_labels):
            r_val = retained[dim] / max_vals[dim]
            c_val = churned[dim]  / max_vals[dim]
            gap   = r_val - c_val
            color = "#E8604C" if gap > 0.15 else "#F5A623" if gap > 0.05 else "#52C4A0"
            st.markdown(
                f'<div style="background:#F7F8FC;border-left:3px solid {color};'
                f'padding:6px 10px;margin-bottom:6px;border-radius:4px;font-size:0.78rem;">'
                f'<b>{dl}</b><br>'
                f'Retained: {r_val:.2f} | Churned: {c_val:.2f}<br>'
                f'Gap: <b style="color:{color}">{gap:+.2f}</b></div>',
                unsafe_allow_html=True)

    # ── KPI summary ──
    st.markdown('<p class="section-title">Retention KPI Summary</p>', unsafe_allow_html=True)
    active_churn   = fdf[fdf["IsActiveMember"]==1]["Exited"].mean()*100 if len(fdf[fdf["IsActiveMember"]==1])>0 else 0
    inactive_churn = fdf[fdf["IsActiveMember"]==0]["Exited"].mean()*100 if len(fdf[fdf["IsActiveMember"]==0])>0 else 0
    multi_churn    = fdf[fdf["NumOfProducts"]>=2]["Exited"].mean()*100  if len(fdf[fdf["NumOfProducts"]>=2])>0 else 0
    single_churn   = fdf[fdf["NumOfProducts"]==1]["Exited"].mean()*100  if len(fdf[fdf["NumOfProducts"]==1])>0 else 0
    card_churn     = fdf[fdf["HasCrCard"]==1]["Exited"].mean()*100      if len(fdf[fdf["HasCrCard"]==1])>0 else 0
    nocard_churn   = fdf[fdf["HasCrCard"]==0]["Exited"].mean()*100      if len(fdf[fdf["HasCrCard"]==0])>0 else 0
    hb_churn       = fdf[fdf["Balance"]>100_000]["Exited"].mean()*100   if len(fdf[fdf["Balance"]>100_000])>0 else 0

    kpi_df = pd.DataFrame({
        "KPI": ["Engagement Retention Ratio","Product Depth Index","Credit Card Stickiness",
                "High-Balance Disengagement","Overall Churn Rate","Avg RSI Score"],
        "Value": [
            f"{active_churn:.1f}% (Active) vs {inactive_churn:.1f}% (Inactive)",
            f"{multi_churn:.1f}% (Multi) vs {single_churn:.1f}% (Single)",
            f"{card_churn:.1f}% (Card) vs {nocard_churn:.1f}% (No Card)",
            f"{hb_churn:.1f}%", f"{churn_rate:.1f}%", f"{avg_rsi:.1f} / 100",
        ],
        "Signal": [
            "🟢 Activity reduces churn"     if active_churn < inactive_churn else "🔴 Activity not protecting",
            "🟢 Multi-product reduces churn" if multi_churn  < single_churn   else "🔴 Depth not protecting",
            "🟢 Card ownership helps"        if card_churn   < nocard_churn   else "🔴 Card not differentiating",
            "🔴 Premium churn risk"          if hb_churn     > churn_rate     else "🟢 Premium customers stable",
            "—", "—",
        ],
    })
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 5 — CHURN ROOT CAUSE  ← NEW TAB
# ══════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="alert-info">
    🔍 <b>Root Cause Analysis:</b> This tab explains <b>why</b> customers churn — going beyond surface rates
    to identify the dominant driver behind each segment's attrition.
    </div>""", unsafe_allow_html=True)

    # ── Row 1: Churn Driver Breakdown + Funnel ──
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="section-title">Primary Churn Driver Breakdown</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">What is the #1 reason customers are leaving?</p>', unsafe_allow_html=True)
        churned_only = fdf[fdf["Exited"] == 1]
        if len(churned_only) > 0:
            driver_counts = churned_only["ChurnDriver"].value_counts().reset_index()
            driver_counts.columns = ["Driver", "Count"]
            driver_counts["Pct"] = (driver_counts["Count"] / len(churned_only) * 100).round(1)
            driver_colors = {
                "Inactivity":          C["coral"],
                "Product Overload":    C["amber"],
                "Low Product Depth":   C["sky"],
                "Age Risk":            C["navy"],
                "Zero Balance":        C["lavender"],
                "Other":               C["slate"],
            }
            fig_d = px.bar(driver_counts, x="Pct", y="Driver", orientation="h",
                color="Driver", color_discrete_map=driver_colors,
                text=driver_counts.apply(lambda r: f"{r['Count']:,}  ({r['Pct']:.1f}%)", axis=1))
            fig_d.update_traces(textposition="outside", showlegend=False)
            fig_d.update_layout(
                xaxis_title="% of Churned Customers", yaxis_title="",
                plot_bgcolor=C["white"], paper_bgcolor=C["white"],
                margin=dict(l=150, r=60, t=10, b=30), height=300,
                font=dict(family="DM Sans", size=12), showlegend=False,
                xaxis=dict(gridcolor="#E4EAF4"),
            )
            st.plotly_chart(bold_theme(fig_d), use_container_width=True, theme=None)
        else:
            st.info("No churned customers in current filter selection.")

    with c2:
        st.markdown('<p class="section-title">Engagement → Churn Funnel</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">How customers move from engaged to churned</p>', unsafe_allow_html=True)
        stages = ["Total Customers", "Active Members", "Multi-Product", "RSI > 65", "Retained"]
        counts = [
            len(fdf),
            fdf["IsActiveMember"].sum(),
            (fdf["NumOfProducts"] >= 2).sum(),
            (fdf["RSI"] > 65).sum(),
            (fdf["Exited"] == 0).sum(),
        ]
        fig_f = go.Figure(go.Funnel(
            y=stages, x=counts,
            textinfo="value+percent initial",
            marker=dict(color=[C["navy"], C["blue"], C["sky"], C["mint"], C["amber"]]),
            connector=dict(line=dict(color=C["ice"], width=2)),
        ))
        fig_f.update_layout(
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            font=dict(family="DM Sans", size=12, color=C["navy"]),
        )
        st.plotly_chart(bold_theme(fig_f), use_container_width=True, theme=None)

    # ── Row 2: Multi-factor breakdown ──
    st.markdown('<p class="section-title">Multi-Factor Churn Breakdown: Age × Geography × Activity</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Combined view of where the highest churn concentrations exist</p>', unsafe_allow_html=True)
    mf = fdf.groupby(["AgeGroup","Geography","IsActiveMember"], observed=True)["Exited"]\
            .agg(["mean","count"]).reset_index()
    mf["ChurnRate"]  = (mf["mean"] * 100).round(1)
    mf["Activity"]   = mf["IsActiveMember"].map({1:"Active", 0:"Inactive"})
    mf["SegmentSize"] = mf["count"]
    fig_mf = px.scatter(
        mf, x="AgeGroup", y="ChurnRate",
        color="Geography", size="SegmentSize",
        symbol="Activity",
        symbol_map={"Active": "circle", "Inactive": "x"},
        color_discrete_map={"France": C["blue"], "Germany": C["coral"], "Spain": C["mint"]},
        hover_data=["SegmentSize", "Activity"],
        labels={"ChurnRate": "Churn Rate (%)", "AgeGroup": "Age Group"},
        size_max=30,
    )
    fig_mf.add_hline(y=churn_rate, line_dash="dot", line_color=C["slate"],
                     annotation_text=f"Avg {churn_rate:.1f}%", annotation_position="right")
    fig_mf.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor=C["white"],
        margin=dict(l=10, r=10, t=10, b=30), height=360,
        font=dict(family="DM Sans", size=12),
        xaxis=dict(gridcolor="#E4EAF4"), yaxis=dict(gridcolor="#E4EAF4"),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(bold_theme(fig_mf), use_container_width=True, theme=None)

    # ── Row 3: Churn Driver Score per Segment ──
    st.markdown('<p class="section-title">Churn Driver Score — Segment Heatmap</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Which factor dominates churn in each engagement profile?</p>', unsafe_allow_html=True)
    seg_driver = fdf[fdf["Exited"]==1].groupby(["EngagementProfile","ChurnDriver"]).size().reset_index(name="Count")
    if len(seg_driver) > 0:
        pivot_sd = seg_driver.pivot_table(index="EngagementProfile", columns="ChurnDriver",
                                          values="Count", fill_value=0)
        fig_sd = px.imshow(pivot_sd,
            color_continuous_scale=[[0, C["ice"]], [0.5, C["sky"]], [1, C["coral"]]],
            text_auto=True, aspect="auto",
            labels={"color": "Churned Customers"})
        fig_sd.update_layout(
            plot_bgcolor=C["white"], paper_bgcolor=C["white"],
            margin=dict(l=200, r=10, t=10, b=80), height=280,
            font=dict(family="DM Sans", size=11),
            xaxis_title="Churn Driver", yaxis_title="",
        )
        st.plotly_chart(bold_theme(fig_sd), use_container_width=True, theme=None)
    else:
        st.info("No churned data available for current filter selection.")

    # ── Root Cause Insight Cards ──
    st.markdown('<p class="section-title">Root Cause Insight Cards</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Plain-language explanations of what\'s driving churn in each segment</p>', unsafe_allow_html=True)
    insight_cols = st.columns(2)
    insights = [
        ("🔴 Inactivity is the #1 Driver",
         f"Inactive members churn at <b>{inactive_churn:.1f}%</b> vs <b>{fdf[fdf['IsActiveMember']==1]['Exited'].mean()*100:.1f}%</b> for active ones. "
         "The bank is not providing enough daily-use value to keep disengaged customers connected.",
         "alert-danger"),
        ("🟡 Age 41–50 is the Critical Window",
         "Middle-aged customers carry the highest financial commitments but also the highest churn rate. "
         "This suggests the bank's products are not meeting the needs of customers at peak financial complexity.",
         "alert-warning"),
        ("🟡 Single-Product Customers are Vulnerable",
         f"<b>{single_churn:.1f}%</b> of single-product holders churn. "
         "Moving them to 2 products reduces churn by ~71%. This is the highest-ROI retention action available.",
         "alert-warning"),
        ("🔴 Germany Needs Market-Level Intervention",
         "Germany shows elevated churn across every segment — not just inactive or single-product holders. "
         "This is a market-level engagement breakdown, not a product issue, and requires a dedicated strategy.",
         "alert-danger"),
    ]
    for i, (title, body, css) in enumerate(insights):
        insight_cols[i % 2].markdown(
            f'<div class="{css}" style="border-radius:10px;padding:14px 16px;margin-bottom:10px;">'
            f'<b>{title}</b><br><span style="font-size:0.81rem">{body}</span></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 6 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="alert-info">
    ℹ️ Use the sidebar filters to narrow the customer base, then explore individual profiles below.
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Filtered Customer Dataset</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-sub">{len(fdf):,} customers match your current filter criteria</p>', unsafe_allow_html=True)

    display_df = fdf[[
        "CustomerId","Surname","Geography","Gender","Age","CreditScore",
        "Balance","NumOfProducts","Tenure","HasCrCard",
        "IsActiveMember","EstimatedSalary","EngagementProfile","RSI","ChurnDriver","Exited"
    ]].copy()
    display_df["Balance"]         = display_df["Balance"].map("€{:,.0f}".format)
    display_df["EstimatedSalary"] = display_df["EstimatedSalary"].map("€{:,.0f}".format)
    display_df["RSI"]             = display_df["RSI"].map("{:.1f}".format)
    display_df["IsActiveMember"]  = display_df["IsActiveMember"].map({1: "✅ Active", 0: "❌ Inactive"})
    display_df["HasCrCard"]       = display_df["HasCrCard"].map({1: "Yes", 0: "No"})
    display_df["Exited"]          = display_df["Exited"].map({0: "Retained", 1: "🚨 Churned"})

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=450)

    csv_out = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Dataset as CSV",
        data=csv_out, file_name="ecb_filtered_customers_v3.csv", mime="text/csv",
    )