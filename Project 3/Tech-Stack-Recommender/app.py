"""
Streamlit Web UI for Tech Stack Recommender
Minimal, clean interface with neutral colors and no animations.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🧠",
    layout="centered"
)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion import get_user_skills
from vectorizer import build_tfidf_vectors
from similarity import score_all_roles
from recommender import recommend, save_log

# Custom CSS for neutral styling
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; }
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .rank-label { font-size: 12px; color: #888888; text-transform: uppercase; letter-spacing: 1px; }
    .role-name { font-size: 20px; font-weight: 700; color: #1A1A1A; margin: 4px 0; }
    .score-label { font-size: 14px; color: #444444; }
    h1, h2, h3 { color: #1A1A1A; }
    .stButton > button {
        background-color: #2C2C2C;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
    }
    .stButton > button:hover { background-color: #444444; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'skills' not in st.session_state:
    st.session_state.skills = ['', '', '']
if 'results' not in st.session_state:
    st.session_state.results = None

# Title
st.title("🧠 Tech Stack Recommender")
st.markdown("""
**Powered by TF-IDF + Cosine Similarity**  
DecodeLabs Internship — Project 3
""")

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data', 'raw_skills.csv')
df = pd.read_csv(csv_path)

# Section: Enter Your Skills
st.header("Enter Your Skills")
st.markdown("Provide at least 3 skills to get recommendations.")

# Dynamic skill input boxes
cols = st.columns(1)
with cols[0]:
    for i in range(len(st.session_state.skills)):
        st.session_state.skills[i] = st.text_input(
            f"Skill {i+1}",
            value=st.session_state.skills[i],
            key=f"skill_{i}",
            placeholder=f"e.g., python"
        )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Add Another Skill", key="add_skill"):
            st.session_state.skills.append('')
            st.rerun()
    
    with col2:
        if len(st.session_state.skills) > 3 and st.button("Remove Last Skill", key="remove_skill"):
            st.session_state.skills.pop()
            st.rerun()

# Get Recommendations button
if st.button("Get Recommendations", type="primary", key="get_recs"):
    # Validate skill count
    filled_skills = [s.strip().lower().replace(" ", "_") for s in st.session_state.skills if s.strip()]
    
    if len(filled_skills) < 3:
        st.warning("⚠️ Please enter at least 3 skills.")
    else:
        # Try to build vectors and get recommendations
        try:
            # Vectorize
            job_matrix, user_vector, vectorizer = build_tfidf_vectors(df, filled_skills)
            
            # Score
            scores = score_all_roles(user_vector, job_matrix)
            
            # Recommend
            results = recommend(df, scores, top_n=3)
            
            # Save log
            output_path = os.path.join(script_dir, 'outputs', 'recommendation_log.txt')
            save_log(filled_skills, results, output_path)
            
            # Store results in session state
            st.session_state.results = results
            st.rerun()
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")

# Display results if available
if st.session_state.results is not None:
    st.divider()
    st.header("Your Top 3 Recommended Career Paths")
    
    results = st.session_state.results
    for _, row in results.iterrows():
        rank = int(row['rank'])
        job_role = row['job_role']
        score = float(row['similarity_score'])
        score_pct = score * 100
        
        st.markdown(f"""
        <div class="result-card">
            <div class="rank-label">Rank {rank}</div>
            <div class="role-name">{job_role}</div>
            <div class="score-label">Match Score: {score_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show skill profile
    st.divider()
    st.subheader("Your Skill Profile")
    filled_skills = [s.strip().lower().replace(" ", "_") for s in st.session_state.skills if s.strip()]
    st.code(", ".join(filled_skills), language="text")
