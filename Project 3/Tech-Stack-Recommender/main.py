"""
CLI Entry Point — Orchestrates all 4 phases with clear console headers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from ingestion import get_user_skills
from vectorizer import build_tfidf_vectors
from similarity import score_all_roles
from recommender import recommend, save_log

def main():
    print("\n" + "=" * 60)
    print("  TECH STACK RECOMMENDER — AI Recommendation Engine")
    print("  DecodeLabs Internship — Project 3")
    print("=" * 60)
    
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data', 'raw_skills.csv')
    
    if not os.path.exists(csv_path):
        print(f"\nError: Dataset not found at {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Phase 1: Ingestion
    print("\n" + "=" * 60)
    print("PHASE 1: USER SKILL INGESTION")
    print("=" * 60)
    user_skills = get_user_skills(min_skills=3)
    
    # Phase 2: Vectorization
    print("\n" + "=" * 60)
    print("PHASE 2: TF-IDF VECTORIZATION")
    print("=" * 60)
    try:
        job_matrix, user_vector, vectorizer = build_tfidf_vectors(df, user_skills)
        print(f"✓ Vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    
    # Phase 3: Similarity Scoring
    print("\n" + "=" * 60)
    print("PHASE 3: COSINE SIMILARITY SCORING")
    print("=" * 60)
    scores = score_all_roles(user_vector, job_matrix)
    print(f"✓ Scoring complete. {len(scores)} job roles evaluated.")
    
    # Phase 4: Recommendation
    print("\n" + "=" * 60)
    print("PHASE 4: RANKING & FILTERING")
    print("=" * 60)
    results = recommend(df, scores, top_n=3)
    print(f"✓ Top 3 recommendations generated.")
    
    # Display Results
    print("\n" + "=" * 60)
    print("TOP 3 RECOMMENDED CAREER PATHS FOR YOUR SKILL PROFILE")
    print("=" * 60)
    for _, row in results.iterrows():
        rank = int(row['rank'])
        job_role = row['job_role']
        score = float(row['similarity_score'])
        print(f"Rank {rank} | {job_role:<25} | Similarity: {score:.4f}")
    print("=" * 60)
    
    # Save Log
    output_path = os.path.join(script_dir, 'outputs', 'recommendation_log.txt')
    save_log(user_skills, results, output_path)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
