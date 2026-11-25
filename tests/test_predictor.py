"""
Automated Test Suite for Student Performance Predictor & Advisory System

Tests:
- Synthetic dataset generation structure and value ranges
- Advisory rule engine output under varying academic conditions
- Model inference pipeline, feature scaling, and binary classification outputs
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student_data_generator import generate_student_data
from student_predictor_gui import generate_academic_advice, load_model


class TestDataGenerator(unittest.TestCase):
    """Test suite for synthetic dataset generation."""

    def setUp(self):
        self.test_csv = os.path.join(os.path.dirname(__file__), "temp_test_data.csv")

    def tearDown(self):
        if os.path.exists(self.test_csv):
            try:
                os.remove(self.test_csv)
            except OSError:
                pass

    def test_generated_dataframe_structure(self):
        df = generate_student_data(filename=self.test_csv, num_students=50)
        expected_cols = [
            'age', 'gender', 'attendance', 'past_grade',
            'test_score', 'assignments', 'extracurricular',
            'GPA', 'pass_fail', 'risk'
        ]
        self.assertEqual(len(df), 50)
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_gpa_bounds_and_labels(self):
        df = generate_student_data(filename=self.test_csv, num_students=50)
        self.assertTrue((df['GPA'] >= 0.0).all())
        self.assertTrue((df['GPA'] <= 4.33).all())
        # Check pass/fail logic
        for _, row in df.iterrows():
            expected = 'pass' if row['GPA'] >= 2.0 else 'fail'
            self.assertEqual(row['pass_fail'], expected)


class TestAdvisoryEngine(unittest.TestCase):
    """Test suite for the rule-based academic advisory logic."""

    def test_failing_student_with_low_attendance_and_grades(self):
        advice = generate_academic_advice(
            prediction="fail",
            attendance=60.0,
            past_grade=45.0,
            test_score=40.0,
            assignments=45.0,
            extracurricular="yes"
        )
        self.assertTrue(any("attendance to at least 75%" in a for a in advice))
        self.assertTrue(any("tutoring or study groups" in a for a in advice))
        self.assertTrue(any("Limit extracurricular" in a for a in advice))

    def test_passing_student_high_attendance(self):
        advice = generate_academic_advice(
            prediction="pass",
            attendance=90.0,
            past_grade=80.0,
            test_score=85.0,
            assignments=88.0,
            extracurricular="yes"
        )
        self.assertTrue(any("Maintain your performance" in a for a in advice))
        # Should not advise reducing extracurriculars when passing
        self.assertFalse(any("Limit extracurricular" in a for a in advice))

    def test_passing_student_marginal_attendance(self):
        advice = generate_academic_advice(
            prediction="pass",
            attendance=70.0,
            past_grade=75.0,
            test_score=75.0,
            assignments=70.0,
            extracurricular="no"
        )
        self.assertTrue(any("improving attendance" in a for a in advice))


class TestModelInference(unittest.TestCase):
    """Test suite for ML model loading and inference."""

    def test_model_loading_and_inference(self):
        ensemble, scaler = load_model()
        self.assertIsNotNone(ensemble)
        self.assertIsNotNone(scaler)

        # High-performing profile: age 20, male(1), attendance 95, past_grade 90, test 95, assign 90, extra yes(1)
        high_student = np.array([[20, 1, 95.0, 90.0, 95.0, 90.0, 1]])
        scaled_high = scaler.transform(high_student)
        pred_high = ensemble.predict(scaled_high)[0]
        self.assertEqual(pred_high.lower(), "pass")

        # Low-performing profile: age 21, female(0), attendance 40, past_grade 35, test 30, assign 35, extra no(0)
        low_student = np.array([[21, 0, 40.0, 35.0, 30.0, 35.0, 0]])
        scaled_low = scaler.transform(low_student)
        pred_low = ensemble.predict(scaled_low)[0]
        self.assertEqual(pred_low.lower(), "fail")


if __name__ == "__main__":
    unittest.main()
