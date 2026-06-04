# 🚢 Titanic Survival Predictor

> **CodSoft Machine Learning Internship — Task 1**
> A premium, interactive Streamlit web application that uses classical ML algorithms to predict passenger survival on the RMS Titanic.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Features](#-live-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [ML Models](#-ml-models)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [App Pages](#-app-pages)
- [Screenshots](#-screenshots)
- [Key Insights](#-key-insights)
- [Author](#-author)

---

## 🧠 Overview

This project builds an end-to-end **machine learning pipeline** on the Titanic dataset to predict whether a passenger survived the disaster. The results are presented through a beautifully designed **Streamlit dashboard** with four interactive pages covering data exploration, model evaluation, and live passenger prediction.

The app was built as **Task 1** of the [CodSoft](https://www.codsoft.in/) Machine Learning Internship.

---

## ✨ Live Features

| Feature | Details |
|---|---|
| 🏠 Overview Page | KPI cards, donut chart, survival-by-class bar chart, dataset preview, key statistical insights |
| 📊 EDA Page | Age distributions, sex vs survival, family size analysis, title-based survival, fare distributions, embarkation crosstab, correlation heatmap |
| 🤖 Model Performance | Accuracy comparison table, confusion matrix, multi-model ROC curves, feature importance chart, full classification report |
| 🎯 Predict Survival | Interactive passenger form, probability bar chart, context-aware explanatory factors |

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.30 | Web application framework |
| `pandas` | ≥ 1.5 | Data loading & manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `scikit-learn` | ≥ 1.2 | ML models, preprocessing, evaluation |
| `matplotlib` | ≥ 3.6 | Custom charts with dark theme |
| `seaborn` | ≥ 0.12 | Heatmaps |

---

## 📁 Project Structure

```
CODSOFT-TASK-1/
│
├── TASK-1/
│   ├── App.py          # Main Streamlit application (851 lines)
│   ├── task1.py        # (placeholder / scratch)
│   ├── task1.ipynb     # Jupyter notebook (exploratory work)
│   └── train.csv       # Titanic training dataset (891 rows)
│
├── LICENSE
└── README.md           # ← You are here
```

---

## 📊 Dataset

**Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger ID |
| `Survived` | Target: 0 = No, 1 = Yes |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Full passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare (£) |
| `Embarked` | Port: C = Cherbourg, Q = Queenstown, S = Southampton |

**Missing value handling:**
- `Age` → filled with **median**
- `Embarked` → filled with **mode**
- `Fare` → filled with **median**

---

## ⚙️ Feature Engineering

The raw data is enriched with the following derived features before modelling:

| New Feature | How It's Created |
|---|---|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `1` if `FamilySize == 1`, else `0` |
| `Title` | Extracted from `Name` using regex; rare titles grouped as `"Rare"` |
| `Sex_enc` | Label-encoded `Sex` |
| `Embarked_enc` | Label-encoded `Embarked` |
| `Title_enc` | Label-encoded `Title` |
| `AgeBand` | `Age` binned into 5 equal-width bands |
| `FareBand` | `Fare` binned into 4 quantile bands |

**Final feature set used for training:**
```
Pclass, Sex_enc, Age, Fare, FamilySize, IsAlone,
Embarked_enc, Title_enc, AgeBand, FareBand
```

---

## 🤖 ML Models

Three classifiers are trained, evaluated, and compared:

| Model | Key Hyperparameters |
|---|---|
| **Random Forest** | `n_estimators=200`, `max_depth=6`, `random_state=42` |
| **Gradient Boosting** | `n_estimators=150`, `max_depth=4`, `random_state=42` |
| **Logistic Regression** | `max_iter=500`, `random_state=42` |

**Evaluation metrics:**
- Test Accuracy
- 5-Fold Cross-Validation Mean ± Std
- Confusion Matrix
- ROC-AUC Curve
- Classification Report (Precision, Recall, F1)
- Feature Importances (from Random Forest)

The **best model** (highest CV mean accuracy) is automatically selected and used for the live prediction page.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/CODSOFT-TASK-1.git
cd CODSOFT-TASK-1/TASK-1
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ Running the App

Make sure you are inside the `TASK-1/` directory (where `App.py` and `train.csv` live):

```bash
cd TASK-1
streamlit run App.py
```

Then open your browser at **http://localhost:8501**

---

## 📱 App Pages

### 🏠 Overview
- Hero banner with project title
- KPI metrics: total passengers, survivors, survival %, best model accuracy
- Donut chart (overall survival) + bar chart (survival by class)
- Dataset snapshot (first 8 rows)
- Three insight cards: Gender effect, Class effect, Children priority

### 📊 Exploratory Data Analysis
Three tabs:

**Demographics Tab**
- Age distribution histogram (survived vs. not)
- Survival count by sex
- Survival rate by family size
- Survival rate by passenger title

**Fare & Class Tab**
- Fare distribution by class
- Stacked bar: embarkation port by class
- Heatmap: survival rate by sex × class

**Feature Correlations Tab**
- Lower-triangle correlation heatmap of all numeric features
- Interpretation guide

### 🤖 Model Performance
- Summary comparison table (test accuracy, CV mean, CV std)
- Interactive model selector for detailed view
- Confusion matrix heatmap
- Multi-model ROC curves with AUC scores
- Random Forest feature importance chart
- Expandable full classification report

### 🎯 Predict Survival
A form to fill in passenger details:
- **Passenger Class** (1st / 2nd / 3rd)
- **Sex**
- **Age** (slider)
- **Siblings / Spouses aboard** (slider)
- **Parents / Children aboard** (slider)
- **Port of Embarkation** (Southampton / Cherbourg / Queenstown)
- **Fare** (number input)
- **Title** (Mr / Mrs / Miss / Master / Rare)

On submit:
- Binary survival prediction with confidence score
- Probability bar chart (Survived vs. Did Not Survive)
- Context-aware factors explaining the prediction

---

## 🔍 Key Insights

| Finding | Statistic |
|---|---|
| Women survival rate | ~74% |
| Men survival rate | ~19% |
| 1st class survival | ~63% |
| 3rd class survival | ~24% |
| Children (≤12 yrs) survival | ~58% |
| Strongest survival predictors | Sex, Passenger Class, Fare, Title |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Atif Khan**
- 🎓 CodSoft Machine Learning Intern
- 💻 Built with ❤️ using Python & Streamlit

---

*Built with [Streamlit](https://streamlit.io/) · CodSoft ML Internship · Task 1*
