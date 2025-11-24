# AIPI-590---XAI-Final-Project-
# Explainable Mental Health Classifier (Streamlit Demo)

This project is a **Responsible AI web app** that demonstrates how explainability techniques can be applied to a synthetic mental‑health text classifier.  
The site allows users to input a sentence and see:
- The predicted category (**help**, **neutral**, **positive**)  
- A LIME-based explanation highlighting which words influenced the prediction  
- Crisis‑indicator detection (e.g., expressions suggesting distress)

The dataset, model design, and explainability approach are intentionally simple and transparent to support ethical analysis, not real‑world diagnosis.

---

## 🚀 Project Overview

This project includes:
- A **synthetic mental‑health dataset** (900 samples)  
- A **DistilBERT classifier** trained on 3 emotional categories  
- A **Streamlit app** for interactive predictions and LIME explanations  
- Responsible AI documentation on dataset design and ethical considerations  

The goal is to explore:
- How dataset choices affect model behavior  
- How LIME explanations reveal feature importance  
- Why synthetic data is safer for sensitive domains  
- The challenges and limitations of explainable NLP tools  

---

## 📦 Installation & Requirements

### 1. Clone the project
```
git clone <your_repo_url>
cd AIPI-590---XAI-Final-Project-
```

### 2. Install Python packages
You can install all required packages with:

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```
streamlit run app.py
```

---

## 📄 `requirements.txt`

The following dependencies are needed to run the project:

```
streamlit
transformers
torch
numpy<2.0
pandas
lime
scikit-learn
sentencepiece
```

> ⚠️ **Note:**  
> `numpy<2.0` ensures compatibility with LIME and prevents SHAP import errors.

---

## 📚 Dataset Description

The classifier is trained on **900 fully synthetic samples**:
- 300 **help**
- 300 **neutral**
- 300 **positive**

The dataset is designed to:
- Avoid any real mental‑health disclosures  
- Enable transparent control over crisis‑indicator phrasing  
- Improve LIME interpretability  
- Avoid ethical risks associated with scraping or using real user text  

The synthetic dataset file is:  
`synthetic_mental_health_dataset_v3_900.csv`

---

## 🧠 Explainability

The app uses **LIME** (Local Interpretable Model-Agnostic Explanations) to show which words contribute most to the model’s prediction.

Users can view:
- Highlighted key phrases  
- Positive/negative contribution of each token  
- Why the model chooses a particular emotional category  

This supports responsible evaluation of model behavior.

---

## ⚠️ Limitations

### ❌ SHAP Integration Not Fully Supported
SHAP currently raises:
```
ImportError: Numba needs NumPy 2.0 or less. Got NumPy 2.2.
```
Even with `numpy<2.0`, other SHAP modules fail due to compatibility issues in Python 3.12.

For this reason, the app **uses LIME instead of SHAP**.

### ⏳ LIME is Slow
- LIME performs **many forward passes** to build local perturbation samples  
- Even with batching enabled, a single explanation may take **3–6 seconds** depending on machine  

This is normal for LIME and expected for transformer models.

### 🚫 Not a Diagnostic Tool
The model:
- Is trained on synthetic text  
- Does **not** reflect real clinical patterns  
- Should **not** be used for any mental‑health assessment  

The purpose is purely educational and to demonstrate responsible AI workflows.

---

## 🏁 Credits
Created by **Sizhe Chen**  
Duke University — Responsible AI Final Project (Fall 2025)