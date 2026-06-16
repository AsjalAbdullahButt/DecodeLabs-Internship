"""
PHASE 3 — Similarity Engine
Computes cosine similarity between user vector and each job role vector.

FORMULA: cos(θ) = (A · B) / (||A|| × ||B||)

WHY COSINE NOT EUCLIDEAN:
- Euclidean is sensitive to vector magnitude (length of description)
- Cosine only measures directional alignment (shared interests)
- A short user profile (3 skills) vs a long job description should still score high
  if the directions align — Euclidean would penalize this, Cosine does not.
"""

import numpy as np

def cosine_similarity_score(user_vector: np.ndarray, item_vector: np.ndarray) -> float:
    """
    Compute cosine similarity between one user vector and one item vector.

    Args:
        user_vector (np.ndarray): 1D user TF-IDF vector
        item_vector (np.ndarray): 1D job role TF-IDF vector

    Returns:
        float: Similarity score between 0 and 1 (since TF-IDF is non-negative)

    Edge Case: If either vector is all zeros, return 0.0 (avoid division by zero)
    """
    # Flatten vectors to 1D if needed
    user_vector = user_vector.flatten()
    item_vector = item_vector.flatten()
    
    # Calculate dot product (numerator)
    dot_product = np.dot(user_vector, item_vector)
    
    # Calculate norms (denominator)
    user_norm = np.linalg.norm(user_vector)
    item_norm = np.linalg.norm(item_vector)
    
    # Handle division by zero
    if user_norm == 0 or item_norm == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (user_norm * item_norm)
    
    return float(similarity)

def score_all_roles(user_vector: np.ndarray, job_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between user vector and ALL job role vectors.

    Args:
        user_vector (np.ndarray): shape (1, vocab_size) or (vocab_size,)
        job_matrix (np.ndarray): shape (n_roles, vocab_size)

    Returns:
        np.ndarray: 1D array of similarity scores, one per job role
    """
    # Flatten user vector to 1D
    user_vector = user_vector.flatten()
    
    # Calculate scores for each job role
    scores = []
    for i in range(job_matrix.shape[0]):
        job_vector = job_matrix[i]
        score = cosine_similarity_score(user_vector, job_vector)
        scores.append(score)
    
    return np.array(scores)
