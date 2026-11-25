# train_model.py
"""
Model Training & Evaluation Pipeline for Student Performance Prediction

Loads synthetic student performance data, performs feature preprocessing,
trains an ensemble classifier (tuned Random Forest + Gradient Boosting via Soft Voting),
evaluates metrics on a held-out test split, and serializes the model and scaler.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- File Paths & Portability ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")


def train():
    # ---------------- Load Dataset ---------------- #
    if not os.path.exists(DATA_PATH):
        print("[INFO] student_data.csv not found. Generating synthetic dataset...")
        from student_data_generator import generate_student_data
        generate_student_data(DATA_PATH)

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ---------------- Encode Categorical Features ---------------- #
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['extracurricular'] = df['extracurricular'].map({'yes': 1, 'no': 0})

    # ---------------- Features and Target ---------------- #
    feature_names = ['age', 'gender', 'attendance', 'past_grade', 'test_score', 'assignments', 'extracurricular']
    X = df[feature_names]
    y = df['pass_fail']

    # ---------------- Feature Selection ---------------- #
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)

    # ---------------- Feature Scaling ---------------- #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # ---------------- Train/Test Split (80/20 Stratified) ---------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # ---------------- Train & Tune Random Forest ---------------- #
    print("Training Random Forest with Grid Search...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print(f"  Best RF Params: {grid.best_params_}")

    # ---------------- Gradient Boosting ---------------- #
    print("Training Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

    # ---------------- Ensemble Voting Classifier (Soft Voting) ---------------- #
    print("Fitting Soft-Voting Ensemble Classifier...")
    ensemble = VotingClassifier([('rf', best_rf), ('gb', gb)], voting='soft')
    ensemble.fit(X_train, y_train)

    # ---------------- Evaluation on Held-Out Test Set ---------------- #
    y_pred = ensemble.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Test Set Evaluation ---")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---------------- Save Model & Scaler ---------------- #
    joblib.dump((ensemble, scaler), MODEL_PATH)
    print(f"[OK] Model artifact successfully saved to: {MODEL_PATH}")
    return ensemble, scaler, test_accuracy


if __name__ == "__main__":
    train()
