# Student Academic Performance Predictor & Advisory System

An end-to-end machine learning prototype and desktop application designed to evaluate academic risk indicators, predict student pass/fail outcomes, and generate actionable, rule-based academic recommendations.

---

## Overview & Purpose

Academic advisors and educators often struggle to identify students at risk of course failure early enough to provide targeted support. 

This prototype explores an applied machine learning approach: combining predictive classification with a transparent, rule-based advisory engine. Rather than outputting an isolated probability score, the system pairs its prediction with concrete, actionable recommendations—such as increasing attendance thresholds, seeking study group support, or temporarily balancing extracurricular commitments.

---

## Key Features

- **Soft-Voting Ensemble Model:** Combines a hyperparameter-tuned `RandomForestClassifier` and a `GradientBoostingClassifier` to output robust class predictions based on soft probability voting.
- **Rule-Based Advisory Engine:** Maps identified risk factors (low attendance, lagging assignment scores, high extracurricular commitments) to constructive intervention advice.
- **Interactive Desktop Interface:** Built with Python's built-in `tkinter` library, providing an accessible graphical interface for entering student metrics and reviewing guidance.
- **Session Feedback Logging:** Records prediction sessions and generated advice to an append-only CSV log for review and auditability.
- **Automated Test Suite:** Built-in unit tests verifying synthetic data generation, advisory rules, and model inference.

---

## Machine Learning Pipeline

```
Synthetic Dataset (student_data.csv)
       │
       ▼ [1. Preprocessing & Encoding]
Categorical mapping (gender, extracurricular)
       │
       ▼ [2. Feature Selection & Standardization]
SelectKBest (f_classif) + StandardScaler
       │
       ▼ [3. Stratified 80/20 Train/Test Split]
X_train (80%) ──┬──► X_test (20% held-out)
                │
       ▼ [4. Model Training & Tuning]
├── Random Forest (GridSearchCV on n_estimators, max_depth)
└── Gradient Boosting (200 estimators)
       │
       ▼ [5. Soft-Voting Ensemble (VotingClassifier)]
Aggregated probability predictions
       │
       ▼ [6. Evaluation on Held-Out Split]
Accuracy, Precision, Recall, F1-Score
       │
       ▼ [7. Serialization]
student_model.pkl (Ensemble Model + Scaler)
```

---

## Dataset Description & Limitations

> **Important Disclosure:** The included dataset (`student_data.csv`) is **entirely synthetic**, generated programmatically via `student_data_generator.py`. 
> 
> It contains **no real-world student records, identifying information, or institution-specific data**.

### Schema Definition
Each synthetic student profile comprises 7 operational features:
- `age`: Integer between 18 and 24.
- `gender`: Binary categorical (`male`, `female`).
- `attendance`: Float percentage (20% to 100%).
- `past_grade`: Float percentage score (15% to 100%).
- `test_score`: Float percentage score (15% to 100%).
- `assignments`: Float percentage score (15% to 100%).
- `extracurricular`: Binary indicator (`yes`, `no`).
- `pass_fail`: Binary target (`pass` if composite GPA $\ge$ 2.0, else `fail`).

---

## Project Structure

```
student-performance-predictor/
├── student_data_generator.py  # Synthetic dataset generator
├── train_model.py             # Feature engineering, training, and evaluation
├── student_predictor_gui.py   # Tkinter desktop GUI and advisory engine
├── student_data.csv           # Synthetic dataset (300 records)
├── student_model.pkl          # Pre-trained model & scaler artifact
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Rules for bytecode and runtime logs
├── tests/
│   └── test_predictor.py     # Automated unit tests
└── README.md                  # Project documentation
```

---

## Installation & Requirements

### Prerequisites
- **Python 3.8+** (Tested on Python 3.10 through 3.14)
- Tkinter (included with standard Python desktop installations)

### Setup
1. Clone or download the repository:
   ```bash
   git clone https://github.com/<your-username>/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launch the Desktop GUI
Run the GUI entry point to open the interactive advisory window:

```bash
python student_predictor_gui.py
```

1. Enter student metrics (Age, Attendance %, Past Grade %, Test Score %, Assignments %).
2. Select Gender and Extracurricular participation.
3. Click **"Predict & Advise"** to view the predicted outcome and tailored academic recommendations.

### 2. Retrain the Machine Learning Model
To retrain the ensemble classifier on the synthetic dataset and update `student_model.pkl`:

```bash
python train_model.py
```

Sample training output:
```text
Loading dataset from: student_data.csv
Training samples: 240 | Test samples: 60
Training Random Forest with Grid Search...
  Best RF Params: {'max_depth': 10, 'n_estimators': 100}
Training Gradient Boosting Classifier...
Fitting Soft-Voting Ensemble Classifier...

--- Test Set Evaluation ---
Test Accuracy: 95.00%

Classification Report:
              precision    recall  f1-score   support
        fail       0.86      1.00      0.92        18
        pass       1.00      0.93      0.96        42
    accuracy                           0.95        60

[OK] Model artifact successfully saved to: student_model.pkl
```

### 3. Generate a Fresh Synthetic Dataset
To regenerate the synthetic training dataset with different random seeds or record counts:

```bash
python student_data_generator.py
```

---

## How the Advisory Logic Works

The advisory engine uses a rule-based heuristic system:
- **Attendance < 75%:** Prompts attendance improvement to ensure lecture comprehension.
- **Academic Scores < 50%:** Flags critically lagging areas and recommends structured tutoring or peer study groups.
- **Extracurricular Overload:** If a failing student participates in extracurriculars, recommends temporary activity reduction to refocus on core academic requirements.
- **Passing Students:** Provides positive reinforcement with continuous attendance monitoring.

*Note: Recommendations are educational heuristics and do not guarantee academic performance.*

---

## Running Automated Tests

Run the unit test suite covering data generation, advisory rules, and model inference:

```bash
python -m unittest discover tests
```

Output:
```text
......
----------------------------------------------------------------------
Ran 6 tests in 3.446s

OK
```

---

## Limitations & Future Enhancements

- **Synthetic Data Limitations:** Because the model was trained on mathematically generated data, it demonstrates technical feasibility rather than real-world predictive validity. Deploying in an academic setting would require calibration against historical, anonymized institutional datasets.
- **Feature Diversity:** Future revisions could incorporate course difficulty indices, credit load, and temporal trends across consecutive semesters.
- **Web Interface:** Migration from Tkinter to a modern web API (e.g. FastAPI) and responsive web dashboard.
