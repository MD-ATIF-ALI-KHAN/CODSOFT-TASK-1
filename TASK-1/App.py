import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS – premium dark-navy theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ── Main background ── */
  .stApp {
      background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a1628 100%);
      color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1526 0%, #101e38 100%);
      border-right: 1px solid rgba(99,179,237,0.15);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
      color: #90cdf4;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
      background: linear-gradient(135deg, rgba(16,30,56,0.9), rgba(20,40,70,0.8));
      border: 1px solid rgba(99,179,237,0.2);
      border-radius: 14px;
      padding: 1rem;
      backdrop-filter: blur(10px);
  }
  [data-testid="metric-container"] label {
      color: #90cdf4 !important;
      font-size: 0.78rem !important;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #f7fafc !important;
      font-size: 2rem !important;
      font-weight: 700;
  }

  /* ── Section headers ── */
  .section-header {
      font-family: 'Outfit', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      color: #90cdf4;
      border-left: 4px solid #4299e1;
      padding-left: 0.75rem;
      margin: 1.5rem 0 1rem;
  }

  /* ── Hero banner ── */
  .hero-banner {
      background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 50%, #1e3a5f 100%);
      border: 1px solid rgba(99,179,237,0.3);
      border-radius: 20px;
      padding: 2.5rem 3rem;
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
  }
  .hero-banner::before {
      content: '';
      position: absolute;
      top: -50%;
      right: -10%;
      width: 400px;
      height: 400px;
      background: radial-gradient(circle, rgba(66,153,225,0.1) 0%, transparent 70%);
      border-radius: 50%;
  }
  .hero-title {
      font-family: 'Outfit', sans-serif;
      font-size: 2.8rem;
      font-weight: 700;
      color: #f7fafc;
      margin: 0;
      line-height: 1.2;
  }
  .hero-subtitle {
      font-size: 1.1rem;
      color: #90cdf4;
      margin-top: 0.5rem;
  }

  /* ── Info box ── */
  .info-box {
      background: rgba(66,153,225,0.08);
      border: 1px solid rgba(99,179,237,0.25);
      border-radius: 12px;
      padding: 1rem 1.25rem;
      margin: 0.75rem 0;
      font-size: 0.9rem;
      color: #cbd5e0;
  }

  /* ── Prediction result ── */
  .pred-survived {
      background: linear-gradient(135deg, rgba(39,103,73,0.5), rgba(47,133,90,0.3));
      border: 2px solid #48bb78;
      border-radius: 16px;
      padding: 1.5rem 2rem;
      text-align: center;
      font-size: 1.5rem;
      font-weight: 700;
      color: #9ae6b4;
  }
  .pred-died {
      background: linear-gradient(135deg, rgba(116,42,55,0.5), rgba(155,44,44,0.3));
      border: 2px solid #fc8181;
      border-radius: 16px;
      padding: 1.5rem 2rem;
      text-align: center;
      font-size: 1.5rem;
      font-weight: 700;
      color: #feb2b2;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      gap: 8px;
      background: rgba(10,14,26,0.5);
      padding: 6px;
      border-radius: 12px;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      border-radius: 8px;
      color: #718096;
      font-weight: 500;
      padding: 0.5rem 1.2rem;
      border: 1px solid transparent;
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
      color: #f7fafc !important;
      border-color: rgba(99,179,237,0.3) !important;
  }

  /* ── Buttons ── */
  .stButton>button {
      background: linear-gradient(135deg, #2b6cb0, #2c5282);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 0.6rem 1.4rem;
      font-weight: 600;
      font-size: 0.95rem;
      transition: all 0.2s ease;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #3182ce, #2b6cb0);
      transform: translateY(-1px);
      box-shadow: 0 4px 15px rgba(66,153,225,0.3);
  }

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] {
      margin-top: 0.5rem;
  }

  /* ── Select box ── */
  .stSelectbox > div > div {
      background: rgba(16,30,56,0.8);
      border: 1px solid rgba(99,179,237,0.25);
      border-radius: 8px;
      color: #e2e8f0;
  }

  /* ── Divider ── */
  hr {
      border-color: rgba(99,179,237,0.15);
  }

  /* ── Chart backgrounds ── */
  .stPlotlyChart, .stPyplot {
      border-radius: 12px;
      overflow: hidden;
  }

  /* ── Number input ── */
  .stNumberInput > div > div > input {
      background: rgba(16,30,56,0.8);
      border: 1px solid rgba(99,179,237,0.25);
      color: #e2e8f0;
      border-radius: 8px;
  }

  /* ── Radio ── */
  .stRadio > div {
      flex-direction: row;
      gap: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
PALETTE = {
    "blue":   "#4299e1",
    "green":  "#48bb78",
    "red":    "#fc8181",
    "yellow": "#f6e05e",
    "purple": "#b794f4",
    "teal":   "#4fd1c5",
    "bg":     "#0a0e1a",
    "card":   "#101e38",
    "text":   "#e2e8f0",
    "muted":  "#718096",
}

MPL_STYLE = {
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    "#2d3748",
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#2d3748",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  PALETTE["card"],
    "legend.edgecolor":  "#2d3748",
}

plt.rcParams.update(MPL_STYLE)


@st.cache_data
def load_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "train.csv")
    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def preprocess(df):
    d = df.copy()
    # Fill missing
    d["Age"].fillna(d["Age"].median(), inplace=True)
    d["Embarked"].fillna(d["Embarked"].mode()[0], inplace=True)
    d["Fare"].fillna(d["Fare"].median(), inplace=True)
    # Feature engineering
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"]    = (d["FamilySize"] == 1).astype(int)
    d["Title"]      = d["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    d["Title"]      = d["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
    )
    d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    le = LabelEncoder()
    d["Sex_enc"]      = le.fit_transform(d["Sex"])
    d["Embarked_enc"] = le.fit_transform(d["Embarked"])
    d["Title_enc"]    = le.fit_transform(d["Title"])
    d["AgeBand"]      = pd.cut(d["Age"], bins=5, labels=False)
    d["FareBand"]     = pd.qcut(d["Fare"], q=4, labels=False)
    features = ["Pclass","Sex_enc","Age","Fare","FamilySize","IsAlone",
                "Embarked_enc","Title_enc","AgeBand","FareBand"]
    return d, features


@st.cache_resource
def train_models(df, features):
    d, _ = preprocess(df)
    X = d[features]
    y = d["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred   = m.predict(X_test)
        y_prob   = m.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(m, X, y, cv=5, scoring="accuracy")
        results[name] = {
            "model":    m,
            "accuracy": accuracy_score(y_test, y_pred),
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "y_test":   y_test,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
        }
    return results, X_test, y_test


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚢 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🎯 Predict Survival"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        '<div class="info-box">This app uses the classic Titanic dataset to train '
        "ML classifiers and predict passenger survival.<br><br>"
        "<b>Dataset:</b> 891 passengers<br>"
        "<b>Features:</b> Age, Sex, Class, Fare, Family…</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#4a5568;font-size:0.75rem;text-align:center;">'
        "Built with Streamlit · CodSoft Internship</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
df_raw = load_data()
df, FEATURES = preprocess(df_raw)
model_results, X_test_global, y_test_global = train_models(df_raw, FEATURES)
best_model_name = max(model_results, key=lambda k: model_results[k]["cv_mean"])
best_model      = model_results[best_model_name]["model"]


# ══════════════════════════════════════════════
# PAGE 1 – Overview
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🚢 Titanic Survival Predictor</p>
        <p class="hero-subtitle">Machine-learning powered analysis of the RMS Titanic disaster</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    survived     = df_raw["Survived"].sum()
    total        = len(df_raw)
    survival_pct = survived / total * 100
    best_acc     = model_results[best_model_name]["accuracy"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passengers", f"{total:,}")
    c2.metric("Survivors", f"{survived:,}", f"{survival_pct:.1f}%")
    c3.metric("Best Model Accuracy", f"{best_acc:.1f}%")
    c4.metric("Best Model", best_model_name.split()[0])

    st.markdown("---")

    # Quick survival breakdown
    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.markdown('<div class="section-header">Survival Rate Overview</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        # Donut
        ax = axes[0]
        vals   = df_raw["Survived"].value_counts().sort_index()
        colors = [PALETTE["red"], PALETTE["green"]]
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=["Did Not Survive", "Survived"],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor=PALETTE["bg"], linewidth=2),
            textprops={"color": PALETTE["text"], "fontsize": 9},
        )
        for at in autotexts:
            at.set_color(PALETTE["bg"])
            at.set_fontweight("bold")
        ax.set_title("Overall", color=PALETTE["teal"], fontsize=11, pad=10)

        # Survival by class
        ax2 = axes[1]
        class_survival = df_raw.groupby("Pclass")["Survived"].mean() * 100
        bars = ax2.bar(
            ["1st", "2nd", "3rd"],
            class_survival.values,
            color=[PALETTE["blue"], PALETTE["yellow"], PALETTE["red"]],
            edgecolor=PALETTE["bg"],
            linewidth=1.5,
            width=0.55,
        )
        for bar, val in zip(bars, class_survival.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=9, fontweight="bold",
            )
        ax2.set_ylim(0, 100)
        ax2.set_title("By Passenger Class", color=PALETTE["teal"], fontsize=11, pad=10)
        ax2.set_ylabel("Survival Rate (%)", fontsize=8)
        ax2.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Dataset Snapshot</div>', unsafe_allow_html=True)
        st.dataframe(
            df_raw[["PassengerId","Survived","Pclass","Name","Sex","Age","Fare"]].head(8),
            use_container_width=True,
            height=260,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        women_surv = df_raw[df_raw["Sex"] == "female"]["Survived"].mean() * 100
        men_surv   = df_raw[df_raw["Sex"] == "male"]["Survived"].mean() * 100
        st.markdown(
            f'<div class="info-box">👩 <b>Women survived at {women_surv:.0f}%</b> vs '
            f'👨 men at {men_surv:.0f}% — reflecting the "women and children first" policy.</div>',
            unsafe_allow_html=True,
        )
    with i2:
        first_surv = df_raw[df_raw["Pclass"] == 1]["Survived"].mean() * 100
        third_surv = df_raw[df_raw["Pclass"] == 3]["Survived"].mean() * 100
        st.markdown(
            f'<div class="info-box">🎩 <b>1st class: {first_surv:.0f}%</b> survival vs '
            f'🥉 3rd class: {third_surv:.0f}% — class privilege was life-saving.</div>',
            unsafe_allow_html=True,
        )
    with i3:
        child_surv = df_raw[df_raw["Age"] <= 12]["Survived"].mean() * 100
        st.markdown(
            f'<div class="info-box">👶 <b>Children (≤12 yrs): {child_surv:.0f}%</b> survival '
            "— youngest passengers had better evacuation priority.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# PAGE 2 – Exploratory Analysis
# ══════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Demographics", "Fare & Class", "Feature Correlations"])

    with tab1:
        c1, c2 = st.columns(2)

        # Age distribution
        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            survived_ages = df_raw[df_raw["Survived"] == 1]["Age"].dropna()
            died_ages     = df_raw[df_raw["Survived"] == 0]["Age"].dropna()
            ax.hist(died_ages,     bins=25, alpha=0.7, color=PALETTE["red"],   label="Did Not Survive", edgecolor="none")
            ax.hist(survived_ages, bins=25, alpha=0.7, color=PALETTE["green"], label="Survived",         edgecolor="none")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution by Survival", fontsize=12, color=PALETTE["teal"])
            ax.legend()
            ax.grid(axis="y")
            st.pyplot(fig)
            plt.close()

        # Sex × Survival
        with c2:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            sex_surv = df_raw.groupby(["Sex", "Survived"]).size().unstack()
            sex_surv.plot(
                kind="bar", ax=ax,
                color=[PALETTE["red"], PALETTE["green"]],
                edgecolor=PALETTE["bg"], linewidth=1.5, width=0.6, rot=0,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.set_title("Survival Count by Sex", fontsize=12, color=PALETTE["teal"])
            ax.legend(["Did Not Survive", "Survived"])
            ax.grid(axis="y")
            st.pyplot(fig)
            plt.close()

        # Family size
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fam_surv = df[df["Survived"].notna()].groupby("FamilySize")["Survived"].mean() * 100
            bars = ax.bar(fam_surv.index, fam_surv.values,
                          color=PALETTE["purple"], alpha=0.85, edgecolor=PALETTE["bg"], linewidth=1.5)
            ax.set_xlabel("Family Size")
            ax.set_ylabel("Survival Rate (%)")
            ax.set_title("Survival Rate by Family Size", fontsize=12, color=PALETTE["teal"])
            ax.grid(axis="y")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{bar.get_height():.0f}%",
                        ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
            st.pyplot(fig)
            plt.close()

        with c4:
            # Title groups
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            title_surv = df[df["Survived"].notna()].groupby("Title")["Survived"].mean().sort_values() * 100
            colors_t   = [PALETTE["blue"] if v >= 50 else PALETTE["red"] for v in title_surv.values]
            bars = ax.barh(title_surv.index, title_surv.values, color=colors_t, edgecolor=PALETTE["bg"])
            ax.set_xlabel("Survival Rate (%)")
            ax.set_title("Survival Rate by Title", fontsize=12, color=PALETTE["teal"])
            ax.axvline(50, color=PALETTE["yellow"], linestyle="--", lw=1, alpha=0.7)
            ax.grid(axis="x")
            st.pyplot(fig)
            plt.close()

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for pclass, col in zip([1, 2, 3], [PALETTE["blue"], PALETTE["yellow"], PALETTE["red"]]):
                fares = df_raw[df_raw["Pclass"] == pclass]["Fare"].dropna()
                ax.hist(fares, bins=20, alpha=0.65, color=col, label=f"Class {pclass}", edgecolor="none")
            ax.set_xlim(0, 300)
            ax.set_xlabel("Fare (£)")
            ax.set_ylabel("Count")
            ax.set_title("Fare Distribution by Class", fontsize=12, color=PALETTE["teal"])
            ax.legend()
            ax.grid(axis="y")
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            class_port = pd.crosstab(df_raw["Pclass"], df_raw["Embarked"])
            class_port.plot(
                kind="bar", ax=ax, stacked=True,
                color=[PALETTE["blue"], PALETTE["green"], PALETTE["yellow"]],
                edgecolor=PALETTE["bg"], linewidth=1.5, rot=0,
            )
            ax.set_xlabel("Passenger Class")
            ax.set_ylabel("Count")
            ax.set_title("Embarkation Port by Class", fontsize=12, color=PALETTE["teal"])
            ax.legend(title="Port", labels=["Cherbourg", "Queenstown", "Southampton"])
            ax.grid(axis="y")
            st.pyplot(fig)
            plt.close()

        # Survival rate heatmap by class × sex
        fig, ax = plt.subplots(figsize=(6, 3))
        pivot = df_raw.pivot_table(
            values="Survived", index="Sex", columns="Pclass", aggfunc="mean"
        ) * 100
        sns.heatmap(
            pivot, annot=True, fmt=".1f",
            cmap="YlOrRd", linewidths=1,
            linecolor=PALETTE["bg"], ax=ax,
            cbar_kws={"label": "Survival %"},
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title("Survival Rate (%) by Sex × Class", fontsize=12, color=PALETTE["teal"])
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 6))
        corr_cols = ["Survived", "Pclass", "Sex_enc", "Age", "SibSp",
                     "Parch", "Fare", "FamilySize", "IsAlone", "AgeBand"]
        corr = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, linecolor=PALETTE["bg"], ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=13, color=PALETTE["teal"], pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="info-box">💡 <b>Reading the heatmap:</b> Values close to <b>+1</b> indicate a '
            "strong positive relationship with survival; values near <b>−1</b> indicate the opposite. "
            "Sex (encoded) and Pclass show the strongest correlations with survival.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# PAGE 3 – Model Performance
# ══════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="section-header">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)

    # Summary table
    rows = []
    for name, r in model_results.items():
        rows.append({
            "Model":        name,
            "Test Accuracy": f"{r['accuracy']*100:.2f}%",
            "CV Mean":       f"{r['cv_mean']*100:.2f}%",
            "CV Std":        f"±{r['cv_std']*100:.2f}%",
            "Best?":         "⭐" if name == best_model_name else "",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")

    # Select model for detail
    sel_name = st.selectbox("Select a model to inspect:", list(model_results.keys()))
    sel      = model_results[sel_name]

    c1, c2 = st.columns(2)

    # Confusion matrix
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(sel["y_test"], sel["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Blues", linewidths=1.5, linecolor=PALETTE["bg"],
            ax=ax, annot_kws={"size": 16, "weight": "bold"},
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix – {sel_name}", color=PALETTE["teal"], fontsize=11)
        st.pyplot(fig)
        plt.close()

    # ROC curve
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, r in model_results.items():
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            roc_auc = auc(fpr, tpr)
            lw = 2.5 if name == sel_name else 1
            ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", color=PALETTE["teal"], fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    # Feature importance (Random Forest only)
    st.markdown("---")
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    rf = model_results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_fi = [PALETTE["blue"] if v >= importances.median() else PALETTE["muted"] for v in importances.values]
    bars = ax.barh(importances.index, importances.values, color=colors_fi, edgecolor=PALETTE["bg"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances", color=PALETTE["teal"], fontsize=12)
    ax.grid(axis="x")
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8, color=PALETTE["text"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification report
    with st.expander("📋 Full Classification Report"):
        report = classification_report(sel["y_test"], sel["y_pred"],
                                        target_names=["Did Not Survive", "Survived"])
        st.code(report)


# ══════════════════════════════════════════════
# PAGE 4 – Predict Survival
# ══════════════════════════════════════════════
elif page == "🎯 Predict Survival":
    st.markdown('<div class="section-header">🎯 Predict Passenger Survival</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Fill in the passenger details below and click <b>Predict</b> '
        f"to see if they would have survived. Using <b>{best_model_name}</b> "
        f"(best model, CV accuracy: {model_results[best_model_name]['cv_mean']*100:.1f}%).</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            pclass = st.selectbox("🎟️ Passenger Class", [1, 2, 3],
                                   format_func=lambda x: f"{x}st" if x == 1 else f"{x}nd" if x == 2 else f"{x}rd")
            sex    = st.selectbox("⚧ Sex", ["male", "female"])
            age    = st.slider("🎂 Age", 0, 80, 28)

        with c2:
            sibsp    = st.slider("👫 Siblings / Spouses aboard", 0, 8, 0)
            parch    = st.slider("👶 Parents / Children aboard", 0, 6, 0)
            embarked = st.selectbox("⚓ Port of Embarkation",
                                     ["S", "C", "Q"],
                                     format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

        with c3:
            fare  = st.number_input("💷 Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
            title = st.selectbox("🏷️ Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

        submitted = st.form_submit_button("🔮 Predict Survival", use_container_width=True)

    if submitted:
        # Build feature vector
        family_size  = sibsp + parch + 1
        is_alone     = int(family_size == 1)
        sex_enc      = 1 if sex == "male" else 0
        embarked_map = {"S": 2, "C": 0, "Q": 1}
        embarked_enc = embarked_map[embarked]
        title_enc    = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}.get(title, 2)
        age_band     = min(int(age / 16), 4)
        fare_band    = 0 if fare < 7.91 else 1 if fare < 14.454 else 2 if fare < 31.0 else 3

        input_df = pd.DataFrame([[
            pclass, sex_enc, age, fare, family_size, is_alone,
            embarked_enc, title_enc, age_band, fare_band,
        ]], columns=FEATURES)

        prediction = best_model.predict(input_df)[0]
        probability = best_model.predict_proba(input_df)[0]

        st.markdown("---")
        st.markdown("### 🔍 Prediction Result")

        col_res, col_prob = st.columns([1.3, 1])
        with col_res:
            if prediction == 1:
                st.markdown(
                    "✅ **SURVIVED**<br>"
                    f"<span style='font-size:0.95rem;color:#9ae6b4;'>This passenger likely would have survived.</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="pred-survived">🛟 Survived<br>'
                    f'<span style="font-size:1rem;font-weight:400;">Confidence: {probability[1]*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "❌ **DID NOT SURVIVE**<br>"
                    f"<span style='font-size:0.95rem;color:#feb2b2;'>This passenger likely would not have survived.</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="pred-died">💀 Did Not Survive<br>'
                    f'<span style="font-size:1rem;font-weight:400;">Confidence: {probability[0]*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

        with col_prob:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            bars = ax.bar(
                ["Did Not Survive", "Survived"],
                [probability[0] * 100, probability[1] * 100],
                color=[PALETTE["red"], PALETTE["green"]],
                edgecolor=PALETTE["bg"], linewidth=1.5, width=0.5,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=11, fontweight="bold",
                )
            ax.set_ylim(0, 110)
            ax.set_ylabel("Probability (%)")
            ax.set_title("Survival Probability", color=PALETTE["teal"], fontsize=11)
            ax.grid(axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Context factors
        st.markdown("### 📌 Key Factors Influencing This Prediction")
        factors = []
        if sex == "female":
            factors.append("✅ Female passengers had a much higher survival rate (74%)")
        else:
            factors.append("⚠️ Male passengers had a lower survival rate (19%)")
        if pclass == 1:
            factors.append("✅ 1st class passengers had the highest survival priority")
        elif pclass == 3:
            factors.append("⚠️ 3rd class passengers had limited access to lifeboats")
        if age <= 12:
            factors.append("✅ Children (≤12) were given evacuation priority")
        if family_size > 4:
            factors.append("⚠️ Large families had lower survival rates")
        elif 2 <= family_size <= 4:
            factors.append("✅ Small family groups tended to survive better")
        else:
            factors.append("ℹ️ Solo travellers had moderate survival rates")

        for f in factors:
            st.markdown(f'<div class="info-box">{f}</div>', unsafe_allow_html=True)
