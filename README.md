# 💓 Heart Disease Prediction App

A **Streamlit-based web app** that predicts the likelihood of **heart disease** based on clinical input data and explains **feature impact using SHAP values**.

---

## 🚀 Features

- 🧠 Predicts presence of heart disease using trained machine learning model
- 📊 Explains each prediction with **SHAP** visualizations (waterfall, bar, fallback table)
- 📌 Supports **sample input** or **manual input**
- ✅ Built-in **error handling** and fallback visualizations
- 🧰 Uses `joblib`, `scikit-learn`, `shap`, `matplotlib`, and `streamlit`

---

## 🗂️ Folder Contents

| File / Folder       | Description                                           |
|---------------------|-------------------------------------------------------|
| `app.py`            | Main Streamlit app                                    |
| `models/`           | Folder with trained model, scaler, SHAP background    |
| └─ `best_model.pkl` | Trained classifier model                              |
| └─ `scaler.pkl`     | Scaler used for preprocessing                         |
| `requirements.txt`  | Python dependencies                                   |
| `README.md`         | You're here!                                          |
| `assets/`           | images of web app                                     |

---

## 🧪 Sample Input Format

The app expects **8 features** as input:

| Feature      | Description                                 |
|--------------|---------------------------------------------|
| `age`        | Age of the patient                          |
| `sex`        | Sex (1 = male, 0 = female)                  |
| `cp`         | Chest pain type (0–3)                       |
| `thalach`    | Maximum heart rate achieved                 |
| `exang`      | Exercise-induced angina (1 = yes, 0 = no)   |
| `oldpeak`    | ST depression induced by exercise           |
| `ca`         | Number of major vessels (0–3) colored       |
| `thal`       | Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible) |

---

## 📈 Model & SHAP Explanation

- **Model**: A classification model (SVC, Random Forest) trained on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Scaling**: Inputs are scaled using `StandardScaler`
- **SHAP**: KernelExplainer is used for SHAP since SVC is not tree-based

---

## ▶️ How to Run

### 🔧 Install Dependencies
#### Optional: create a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### Install dependencies
```
pip install -r requirements.txt
```
#### ▶️ Step 3: Launch the Web App
```
streamlit run app.py
```
The app will open in your default web browser (usually at http://localhost:8501).


### 📜 License

This project is for educational and research purposes. Feel free to use and modify.

### 🙌 Acknowledgments

UCI Heart Disease Dataset
SHAP (SHapley Additive exPlanations)
Streamlit community