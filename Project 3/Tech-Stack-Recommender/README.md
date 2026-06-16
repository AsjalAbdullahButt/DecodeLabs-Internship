# Tech Stack Recommender — AI Recommendation Engine

A content-based filtering recommendation system that suggests the best career paths based on your technical skills.

**DecodeLabs Internship — Project 3**

---

## Overview

The Tech Stack Recommender uses **TF-IDF vectorization** and **Cosine Similarity** to match your skills profile against 15+ job roles and recommend the top 3 career paths that align with your expertise.

### Key Features

- **Phase 1 (Ingestion)**: Capture your technical skills with input validation
- **Phase 2 (Vectorization)**: Convert skills into TF-IDF weighted vectors
- **Phase 3 (Similarity Scoring)**: Compute cosine similarity between your profile and job roles
- **Phase 4 (Recommendation)**: Rank and filter to Top 3 recommendations
- **Dual Interface**: CLI terminal app + Streamlit web UI
- **Persistent Logging**: All recommendations saved to recommendation_log.txt

---

## Project Structure

```
Tech-Stack-Recommender/
├── data/
│   └── raw_skills.csv           # 15 job roles with skill datasets
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # User skill input & validation
│   ├── vectorizer.py            # TF-IDF vectorization
│   ├── similarity.py            # Cosine similarity engine
│   └── recommender.py           # Ranking & logging
├── outputs/
│   └── recommendation_log.txt   # Session logs
├── app.py                       # Streamlit web UI
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone/navigate to the project directory:
   ```bash
   cd Project\ 3/Tech-Stack-Recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### CLI Mode (Terminal)

Run the command-line interface:

```bash
python main.py
```

**Interactive flow:**
1. Enter minimum 3 technical skills (one per line)
2. Type `done` when finished
3. View your Top 3 recommended career paths
4. Results automatically saved to `outputs/recommendation_log.txt`

**Example:**
```
Enter your skills one by one (minimum 3). Type 'done' when finished.
Skill: python
Skill added: python
Skill: sql
Skill added: sql
Skill: machine_learning
Skill added: machine_learning
Skill: done

Profile captured: ['python', 'sql', 'machine_learning']

...

TOP 3 RECOMMENDED CAREER PATHS FOR YOUR SKILL PROFILE
============================================================
Rank 1 | Data Scientist          | Similarity: 0.8923
Rank 2 | Machine Learning Eng.   | Similarity: 0.7541
Rank 3 | Data Engineer           | Similarity: 0.6102
============================================================
Results saved to: outputs/recommendation_log.txt
```

### Web UI Mode (Streamlit)

Run the web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Clean, minimal interface with neutral colors
- Dynamic skill input (add/remove skill fields)
- Real-time recommendations
- Match score percentage for each role
- Skill profile display

---

## Dataset: Job Roles

The system evaluates these 15 job roles:

1. **Data Scientist** — Python, SQL, ML, Statistics, TensorFlow
2. **Backend Developer** — Python, Java, Databases, APIs, Docker
3. **Frontend Developer** — JavaScript, HTML, CSS, React, TypeScript
4. **DevOps Engineer** — AWS, Docker, Kubernetes, Linux, Terraform
5. **Full Stack Developer** — JavaScript, Python, React, Django, APIs
6. **Machine Learning Engineer** — Python, TensorFlow, PyTorch, NLP
7. **Data Engineer** — Python, SQL, Spark, Hadoop, ETL, Airflow
8. **Cloud Architect** — AWS, Azure, GCP, Kubernetes, Terraform
9. **Cybersecurity Analyst** — Linux, Networking, Ethical Hacking, Cryptography
10. **Mobile Developer** — Swift, Kotlin, React Native, Flutter, Firebase
11. **AI Research Scientist** — Python, Deep Learning, NLP, Mathematics
12. **Database Administrator** — SQL, PostgreSQL, MySQL, MongoDB, Indexing
13. **Systems Engineer** — Linux, C/C++, Embedded Systems, Assembly
14. **Blockchain Developer** — Solidity, Ethereum, Web3, Smart Contracts
15. **QA Engineer** — Testing, Automation, Selenium, CI/CD, Agile

---

## How It Works

### Phase 1: Ingestion
- Capture user skills via CLI or web input
- Validate minimum 3 skills
- Sanitize input: lowercase, replace spaces with underscores, deduplicate

### Phase 2: Vectorization (TF-IDF)
- Fit `TfidfVectorizer` on all job role skill descriptions
- Transform job roles into TF-IDF weighted vectors
- Transform user skills using the same vocabulary space
- **Cold Start Guard**: Raise error if user vector is all zeros (no matched skills)

### Phase 3: Similarity Scoring (Cosine Similarity)
- Implement cosine similarity formula manually:
  ```
  cos(θ) = (A · B) / (||A|| × ||B||)
  ```
- Compute score between user vector and each job role vector
- Scores range [0, 1] (non-negative TF-IDF vectors)

### Phase 4: Recommendation
- Sort job roles by similarity score (descending)
- Return Top 3 roles
- Log results with timestamp to `outputs/recommendation_log.txt`

---

## Example Recommendations

**Input:** `python, sql, machine_learning`

**Output:**
```
Rank 1 | Data Scientist          | Similarity: 0.8923
Rank 2 | Machine Learning Eng.   | Similarity: 0.7541
Rank 3 | Data Engineer           | Similarity: 0.6102
```

**Why?** Data Scientist role contains all three user skills (python, sql, machine_learning) with high TF-IDF weights.

---

## Recommendation Log

Each session is logged to `outputs/recommendation_log.txt`:

```
============================================================
Session: 2026-06-16 14:32:00
User Skills: python, sql, machine_learning
------------------------------------------------------------
Rank 1: Data Scientist          | Score: 0.8923
Rank 2: Machine Learning Eng... | Score: 0.7541
Rank 3: Data Engineer           | Score: 0.6102
============================================================
```

---

## Key Design Decisions

### Why TF-IDF?
- Captures skill importance in each job role
- Rare skills weighted higher (better discrimination)
- Compared to binary vectors which treat all skills equally

### Why Cosine Similarity?
- Measures directional alignment, not magnitude
- A short user profile (3 skills) can score high against longer job descriptions
- Handles sparse vectors well (many zero components)

### Why Manual Cosine Implementation?
- Educational: Shows the math clearly
- Transparent: No hidden assumptions in black-box libraries
- Efficient: Direct numpy computation without sklearn overhead

---

## Error Handling

**Cold Start (No Matching Skills):**
```
✗ Error: None of your skills matched the dataset vocabulary. 
Try skills like: python, sql, javascript, aws, docker
```

**Insufficient Skills:**
```
⚠️ Please enter at least 3 skills.
```

**Missing Dataset:**
```
Error: Dataset not found at Project 3/Tech-Stack-Recommender/data/raw_skills.csv
```

---

## Testing Checklist

- [x] `python main.py` runs without errors
- [x] CLI accepts 3+ skills and displays Top 3 recommendations
- [x] Scores are in range [0, 1]
- [x] `outputs/recommendation_log.txt` is created and appended
- [x] Cold start error message shown for unmatched skills (not crash)
- [x] `streamlit run app.py` launches without errors
- [x] Web UI displays results after clicking "Get Recommendations"
- [x] Web UI shows warning if fewer than 3 skills
- [x] Web UI shows error if no skills match vocabulary
- [x] Entering `python sql machine_learning` returns Data Scientist as Rank 1

---

## Dependencies

- **streamlit** (1.32.0) — Web UI framework
- **scikit-learn** (1.3.2) — ML utilities (TfidfVectorizer)
- **pandas** (2.1.3) — Data manipulation
- **numpy** (1.26.2) — Numerical computing

---

## Author

DecodeLabs Internship — Project 3  
Built by: AI Engineering Team

---

## License

Internal Project — DecodeLabs
