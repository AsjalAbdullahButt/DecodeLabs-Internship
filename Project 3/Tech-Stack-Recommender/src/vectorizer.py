"""
PHASE 2 — Vectorizer Module
Converts user skill list and job role skill strings into TF-IDF weighted vectors.

CRITICAL RULES:
- Use sklearn.feature_extraction.text.TfidfVectorizer
- Fit the vectorizer on the FULL job role corpus (all skills strings combined)
- Transform BOTH job role strings AND user skill string using the SAME fitted vectorizer
- This ensures user vector and item vectors exist in the SAME vocabulary space
- vocabulary mismatch = similarity math fails silently — prevent this
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def build_tfidf_vectors(df: pd.DataFrame, user_skills: list):
    """
    Args:
        df (pd.DataFrame): Job roles dataframe with 'job_role' and 'skills' columns
        user_skills (list): Cleaned list of user skills from ingestion

    Returns:
        job_matrix (np.ndarray): TF-IDF matrix for all job roles (shape: n_roles x vocab)
        user_vector (np.ndarray): TF-IDF vector for user profile (shape: 1 x vocab)
        vectorizer (TfidfVectorizer): Fitted vectorizer instance (for inspection/debugging)

    IMPLEMENTATION STEPS:
    1. Fit TfidfVectorizer on df['skills'] corpus
    2. Transform df['skills'] → job_matrix
    3. Join user_skills into a single space-separated string → user_string
    4. Transform user_string using the ALREADY FITTED vectorizer (do NOT refit)
    5. Return all three
    
    Cold Start Guard:
    - After transforming user vector, check if it is all zeros
    - If so, raise a ValueError with message:
      "None of your skills matched the dataset vocabulary. 
       Try skills like: python, sql, javascript, aws, docker"
    """
    
    # Step 1: Fit TfidfVectorizer on job roles corpus
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    vectorizer.fit(df['skills'])
    
    # Step 2: Transform job roles to TF-IDF matrix
    job_matrix = vectorizer.transform(df['skills']).toarray()
    
    # Step 3: Join user skills into space-separated string
    user_string = " ".join(user_skills)
    
    # Step 4: Transform user string with ALREADY FITTED vectorizer
    user_vector = vectorizer.transform([user_string]).toarray()
    
    # Step 5: Cold start guard - check if user vector is all zeros
    if np.allclose(user_vector, 0):
        raise ValueError(
            "None of your skills matched the dataset vocabulary. "
            "Try skills like: python, sql, javascript, aws, docker"
        )
    
    return job_matrix, user_vector, vectorizer
