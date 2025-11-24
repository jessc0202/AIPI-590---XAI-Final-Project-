# AIPI-590---XAI-Final-Project-
# Explainable Mental Health Classifier (Streamlit Demo)

This project is a **Responsible AI web app** that demonstrates how explainability techniques can be applied to a synthetic mental‑health text classifier.  
The site allows users to input a sentence and see:
- The predicted category (**help** 🆘, **neutral** 😐, **positive** 🙂)  
- A LIME-based explanation highlighting which words influenced the prediction  
- Crisis‑indicator detection (e.g., expressions suggesting distress or potential self-harm)

⚠️ The dataset, model design, and explainability approach are intentionally simple and transparent to support ethical analysis, not real‑world diagnosis.

---

## Motivation: Building Safer AI Chatbots

As AI chatbots become increasingly common in healthcare and mental support, the need for **trust and transparency** is critical. A "Black Box" model that outputs a prediction without explanation poses significant risks in sensitive domains.

This project addresses two key challenges in building mental health AI:

### 1. The "Black Box" Problem
Standard deep learning models (like BERT) are powerful but opaque. If a user expresses subtle distress and the model ignores it, we need to know **why**. 
* **Solution:** We use **LIME** to peel back the layers, showing exactly which words (e.g., "tired", "end") triggered a specific classification.

### 2. Safety & Crisis Detection
A general-purpose chatbot might miscode language used in crises.
* **Solution:** We implement a **deterministic safety override**. If specific patterns (regex) are detected, the system bypasses the AI model entirely to prioritize safety—a crucial architecture pattern for any responsible AI chatbox.

### 3. The Data Trade-Off (Real vs. Synthetic)
Training on real mental health posts (like Reddit data) is high-performance but ethically risky due to privacy concerns.
* **My Experiment:** I initially attempted to train purely on **Synthetic Data** (also included in the repo called `synthetic_mental_health_dataset.csv`) to maximize privacy. However, I found that synthetic data lacked the linguistic diversity needed for a robust chatbot. Hence, it will always give false predictions.
* **The Solution:** The final deployed model uses the **GoEmotions dataset** (Google Research) as a high-quality, open-source benchmark, while the synthetic dataset is preserved in the repo to demonstrate the "privacy-first" approach we attempted.
---

## 🚀 Project Overview

This project includes:
- A **synthetic mental‑health dataset** (300 samples)  
- A **DistilBERT classifier** trained on 3 emotional categories  
- A **Streamlit app** for interactive predictions and LIME explanations  
- Responsible AI documentation on dataset design and ethical considerations  

The goal is to explore:
- How LIME explanations reveal feature importance  
- The challenges and limitations of explainable NLP tools  

---
## 📂 Project Structure

* `app.py`: The main Streamlit application file. Run this to launch the interface.
* `GoEmotions_Training_Notebook.ipynb`: Jupyter notebook used to fine-tune DistilBERT on the GoEmotions dataset.
* `GoEmotions_Inference_LIME.ipynb`: Notebook demonstrating how LIME works on specific examples (useful for debugging model behavior).
* `synthetic_mental_health_dataset.csv`: The custom synthetic dataset generated for the initial (and educational) privacy experiment.
* `requirements.txt`: List of all Python dependencies required to run the app.

---

## 📦 Installation & Requirements

### 1. Clone the project
```bash
git clone [https://github.com/jessc0202/aipi-590---xai-final-project-.git](https://github.com/jessc0202/aipi-590---xai-final-project-.git)
cd aipi-590---xai-final-project-
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
tokenizers
huggingface-hub
lime
regex
numpy
scikit-learn
```

---

## 📚 Dataset Description

### Primary Training Data: GoEmotions
The deployed model is trained on the **GoEmotions** dataset (Google Research), simplified into 3 distinct categories (from 27 categories) for mental health triage:

* **Help:** (mapped from *sadness, fear, grief, etc.*)
* **Positive:** (mapped from *joy, love, admiration, etc.*)
* **Neutral:** (mapped from *neutral, curiosity, realization*)

### Experimental Data: Synthetic Dataset
We also generated a **Synthetic Mental Health Dataset** (`synthetic_mental_health_dataset.csv`) to test if we could train a model without using any real human data.

* **Outcome:** The synthetic data proved too repetitive and simple for real-world generalization, highlighting the difficulty of purely synthetic training for sensitive domains. It is included in the repo for educational comparison.
---

## 🧠 Explainability

The app uses **LIME** (Local Interpretable Model-Agnostic Explanations) to show which words contribute most to the model’s prediction.

Users can view:
- Highlighted key phrases  
- Positive/negative contribution of each token  
- Why does the model choose a particular emotional category  

This supports responsible evaluation of model behavior.

---

## ⚠️ Limitations

### ⏳ LIME Explanation part is Slow
- LIME performs **many forward passes** to build local perturbation samples  
- Even with batching enabled, a single explanation may take **3–6 seconds** depending on the machine  


### 🚫 Not a Diagnostic Tool
The model:
- Is trained on synthetic text  
- Does **not** reflect real clinical patterns  
- Should **not** be used for any mental‑health assessment  

The purpose is purely educational and to demonstrate responsible AI workflows.

---
Created by **Sizhe Chen**  
Duke University —  Explainable AI Final Project (Fall 2025)
