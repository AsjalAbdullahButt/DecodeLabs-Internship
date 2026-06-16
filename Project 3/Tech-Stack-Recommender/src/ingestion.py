"""
PHASE 1 — Ingestion Module
Captures user skill input and validates minimum data density.
"""

def get_user_skills(min_skills: int = 3) -> list:
    """
    Prompt user to enter skills one by one.
    Enforce minimum of min_skills inputs.
    Sanitize each input: lowercase + strip + replace spaces with underscore.
    Return a list of cleaned skill strings.

    Cold Start Guard:
    - If user enters fewer than min_skills, keep prompting.
    - If user enters a skill already entered, skip it (no duplicates).
    - Empty inputs are silently ignored.

    Args:
        min_skills (int): Minimum number of skills required (default: 3)

    Returns:
        list: Cleaned, deduplicated skill list
    """
    skills = []
    
    print('Enter your skills one by one (minimum 3). Type \'done\' when finished.')
    
    while True:
        user_input = input("Skill: ").strip().lower().replace(" ", "_")
        
        # Check for done command
        if user_input == "done":
            if len(skills) < min_skills:
                print(f"You need at least {min_skills} skills. Please add more.")
                continue
            else:
                break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Check for duplicates
        if user_input in skills:
            print("Skill already added, skipping.")
            continue
        
        # Add skill
        skills.append(user_input)
        print(f"Skill added: {user_input}")
    
    print(f"\nProfile captured: {skills}")
    return skills
