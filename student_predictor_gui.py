# student_predictor_gui.py
"""
Student Academic Performance Predictor & Advisory System (Desktop GUI)

Provides an interactive Tkinter desktop interface for predicting student
pass/fail outcomes and generating actionable, rule-based academic guidance.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import joblib

# ---------------- File Paths & Portability ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
FEEDBACK_PATH = os.path.join(BASE_DIR, "predictions_feedback.csv")


# ---------------- Advisory Rule Engine ---------------- #
def generate_academic_advice(prediction, attendance, past_grade, test_score, assignments, extracurricular):
    """
    Generate tailored, rule-based academic guidance based on student indicators.
    
    Advisory Notice: These recommendations are educational heuristics designed
    to support student reflection and should not be treated as guaranteed outcomes.
    """
    advice_list = []
    is_fail = str(prediction).lower() == "fail"

    if is_fail:
        if attendance < 75:
            advice_list.append("Increase attendance to at least 75%.")
        if past_grade < 50 or test_score < 50 or assignments < 50:
            advice_list.append("Consider joining tutoring or study groups.")
        if str(extracurricular).lower() == "yes":
            advice_list.append("Limit extracurricular activities temporarily.")
    else:
        advice_list.append("Maintain your performance and attendance!")
        if attendance < 75:
            advice_list.append("Try improving attendance for better results.")

    if not advice_list:
        advice_list.append("Continue monitoring course requirements and maintaining balanced study habits.")

    return advice_list


# ---------------- Load Pre-Trained Model ---------------- #
def load_model(path=MODEL_PATH):
    """Load the serialized ensemble model and feature scaler."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pre-trained model artifact not found at '{path}'. "
            f"Please run 'python train_model.py' first to train and generate the model artifact."
        )
    return joblib.load(path)


# ---------------- GUI Setup & Main Loop ---------------- #
def launch_gui():
    try:
        ensemble, scaler = load_model()
    except Exception as e:
        # If running in environment without display or missing model, notify cleanly
        print(f"Error loading model: {e}")
        return

    root = tk.Tk()
    root.title("Student Performance Predictor & Advisor")
    root.geometry("440x780")
    root.configure(bg="#f2e6ff")  # Soft purple background

    title = tk.Label(
        root,
        text="Student Performance Predictor & Advisor",
        font=("Arial", 14, "bold"),
        bg="#f2e6ff",
        fg="#4b0082"
    )
    title.pack(pady=10)

    subtitle = tk.Label(
        root,
        text="Heuristic Academic Advisory Prototype",
        font=("Arial", 9, "italic"),
        bg="#f2e6ff",
        fg="#6a4c93"
    )
    subtitle.pack(pady=0)

    frame = tk.Frame(root, bg="#f2e6ff")
    frame.pack(pady=10)

    def create_label_entry(label_text):
        tk.Label(frame, text=label_text, bg="#f2e6ff", fg="#4b0082").pack()
        entry = tk.Entry(frame)
        entry.pack(pady=3)
        return entry

    # Input Fields
    age_entry = create_label_entry("Age:")
    attendance_entry = create_label_entry("Attendance (0-100%):")
    past_grade_entry = create_label_entry("Past Grade (0-100%):")
    test_score_entry = create_label_entry("Test Score (0-100%):")
    assignments_entry = create_label_entry("Assignments (0-100%):")

    # Categorical Dropdowns
    tk.Label(frame, text="Gender:", bg="#f2e6ff", fg="#4b0082").pack()
    gender_var = tk.StringVar(value="male")
    tk.OptionMenu(frame, gender_var, "male", "female").pack(pady=2)

    tk.Label(frame, text="Extracurricular Participation:", bg="#f2e6ff", fg="#4b0082").pack()
    extra_var = tk.StringVar(value="yes")
    tk.OptionMenu(frame, extra_var, "yes", "no").pack(pady=2)

    # Result & Advisory Display
    result_text = tk.StringVar()
    tk.Label(root, textvariable=result_text, font=("Arial", 12, "bold"), bg="#f2e6ff").pack(pady=10)

    advice_text = tk.StringVar()
    tk.Label(root, textvariable=advice_text, font=("Arial", 10), bg="#f2e6ff", wraplength=380, justify="left").pack(pady=10)

    # Prediction Action
    def predict_student():
        try:
            age = float(age_entry.get())
            attendance = float(attendance_entry.get())
            past_grade = float(past_grade_entry.get())
            test_score = float(test_score_entry.get())
            assignments = float(assignments_entry.get())
            gender_encoded = 1 if gender_var.get() == "male" else 0
            extra_encoded = 1 if extra_var.get() == "yes" else 0

            # Scale and predict
            features = np.array([[age, gender_encoded, attendance, past_grade, test_score, assignments, extra_encoded]])
            features_scaled = scaler.transform(features)
            prediction = ensemble.predict(features_scaled)[0]

            result_text.set(f"Predicted Outcome: {prediction.upper()}")

            # Generate advice
            advice_list = generate_academic_advice(
                prediction, attendance, past_grade, test_score, assignments, extra_var.get()
            )
            advice_text.set("Actionable Academic Guidance:\n- " + "\n- ".join(advice_list))

            # Record feedback session
            feedback = {
                "age": [age],
                "gender": [gender_var.get()],
                "attendance": [attendance],
                "past_grade": [past_grade],
                "test_score": [test_score],
                "assignments": [assignments],
                "extracurricular": [extra_var.get()],
                "prediction": [prediction],
                "advice": ["; ".join(advice_list)]
            }
            df_feedback = pd.DataFrame(feedback)
            try:
                existing = pd.read_csv(FEEDBACK_PATH)
                df_feedback = pd.concat([existing, df_feedback], ignore_index=True)
            except FileNotFoundError:
                pass
            df_feedback.to_csv(FEEDBACK_PATH, index=False)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for age and percentage scores.")

    tk.Button(
        root,
        text="Predict & Advise",
        command=predict_student,
        bg="#800080",
        fg="white",
        padx=12,
        pady=6,
        font=("Arial", 10, "bold")
    ).pack(pady=12)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
