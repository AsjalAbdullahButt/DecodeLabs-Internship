"""
PHASE 4 — Recommender Module
Implements the 4-step ranking pipeline:
Step 1: Ingestion    → already done (passed in as argument)
Step 2: Scoring      → call similarity engine on all roles
Step 3: Sorting      → sort scores descending
Step 4: Filtering    → truncate to Top-N

Also saves results to outputs/recommendation_log.txt
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def recommend(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Steps 3 & 4 of the pipeline: Sort and Filter.

    Args:
        df (pd.DataFrame): Job roles dataframe
        scores (np.ndarray): Cosine similarity scores (one per role)
        top_n (int): Number of top recommendations to return

    Returns:
        pd.DataFrame: Top-N roles with columns ['rank', 'job_role', 'similarity_score']
        Scores formatted to 4 decimal places.
        Sorted descending by similarity_score.
    """
    # Create results dataframe with scores
    results_df = pd.DataFrame({
        'job_role': df['job_role'].values,
        'similarity_score': scores
    })
    
    # Sort descending by similarity score
    results_df = results_df.sort_values('similarity_score', ascending=False).reset_index(drop=True)
    
    # Take top N
    results_df = results_df.head(top_n).reset_index(drop=True)
    
    # Add rank column (1-indexed)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Reorder columns
    results_df = results_df[['rank', 'job_role', 'similarity_score']]
    
    return results_df

def save_log(user_skills: list, results: pd.DataFrame, output_path: str) -> None:
    """
    Append this session's results to recommendation_log.txt.

    Format:
    ============================================================
    Session: 2026-06-15 14:32:00
    User Skills: python, sql, machine_learning
    ------------------------------------------------------------
    Rank 1: Data Scientist          | Score: 0.8923
    Rank 2: Machine Learning Eng... | Score: 0.7541
    Rank 3: Data Engineer           | Score: 0.6102
    ============================================================
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format skills string
    skills_str = ", ".join(user_skills)
    
    # Build log entry
    log_entry = "=" * 60 + "\n"
    log_entry += f"Session: {timestamp}\n"
    log_entry += f"User Skills: {skills_str}\n"
    log_entry += "-" * 60 + "\n"
    
    # Add each recommendation
    for _, row in results.iterrows():
        rank = int(row['rank'])
        job_role = row['job_role']
        score = float(row['similarity_score'])
        log_entry += f"Rank {rank}: {job_role:<25} | Score: {score:.4f}\n"
    
    log_entry += "=" * 60 + "\n\n"
    
    # Append to file
    with open(output_path, 'a') as f:
        f.write(log_entry)
