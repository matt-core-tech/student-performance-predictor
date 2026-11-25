# student_data_generator.py
"""
Synthetic Dataset Generator for Student Performance Prediction

Generates synthetic student performance records for educational, testing,
and prototype modeling purposes.

Notice: This generator produces synthetic mock distributions. The resulting
records do not represent real-world students, personal data, or validated empirical cohorts.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")


def generate_student_data(filename=DEFAULT_DATA_PATH, num_students=300):
    """
    Generate synthetic student performance data with realistic variance.
    
    Each synthetic record includes:
    - age: Integer between 18 and 24
    - gender: 'male' or 'female'
    - attendance: Float between 20% and 100%
    - past_grade: Float between 15% and 100%
    - test_score: Float between 15% and 100%
    - assignments: Float between 15% and 100%
    - extracurricular: 'yes' or 'no'
    - GPA: Synthetic composite index (0.0 to 4.33)
    - pass_fail: 'pass' if GPA >= 2.0 else 'fail'
    - risk: 'low' (GPA >= 3.0), 'medium' (GPA >= 2.0), or 'high'
    """
    np.random.seed(42)

    data = {
        'age': np.random.randint(18, 25, size=num_students),
        'gender': np.random.choice(['male', 'female'], size=num_students),
        'attendance': np.random.uniform(20, 100, size=num_students),
        'past_grade': np.random.uniform(15, 100, size=num_students),
        'test_score': np.random.uniform(15, 100, size=num_students),
        'assignments': np.random.uniform(15, 100, size=num_students),
        'extracurricular': np.random.choice(['yes', 'no'], size=num_students),
    }

    df = pd.DataFrame(data)

    # Derived composite synthetic GPA (0–4.33 scale)
    df['GPA'] = (
        0.3 * df['past_grade'] + 0.4 * df['test_score'] + 0.2 * df['assignments'] + 0.1 * df['attendance']
    ) / 25
    df['GPA'] = df['GPA'].clip(0, 4.33)

    # Pass/fail classification threshold at 2.0 GPA
    df['pass_fail'] = df['GPA'].apply(lambda x: 'pass' if x >= 2.0 else 'fail')

    # Qualitative risk category
    df['risk'] = df['GPA'].apply(lambda x: 'low' if x >= 3.0 else ('medium' if x >= 2.0 else 'high'))

    df.to_csv(filename, index=False)
    print(f"[OK] Synthetic dataset saved to: {filename} ({len(df)} records)")
    return df


if __name__ == "__main__":
    generate_student_data()
