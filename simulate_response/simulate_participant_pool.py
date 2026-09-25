#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File    :   simulate_participant_pool.py
@Time    :   2025/05/15 15:50:53
@Author  :   Manlu Ouyang
@Version :   1.0
@Contact :   mo2615@stern.nyu.edu
@Desc    :   None
'''

import pandas as pd
import random
import uuid
import os

# === Step 1: Define allowed race categories ===
allowed_races = {
    "White": "White",
    "Black or African American": "Black",
    "Asian and Native Hawaiian and Other Pacific Islander": "Asian",
    "American Indian and Alaska Native": "Indigenous",
    "Hispanic or Latino": "Latino"
}

allowed_genders = ["Male", "Female"]

def prompt_with_default(prompt: str, default, caster):
    """Prompt user with a default; empty input returns default."""
    raw = input(f"{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    try:
        return caster(raw)
    except Exception:
        print("Invalid input. Using default.")
        return default


# === Step 2: Get yes/no for race distribution ===
print("Do you want a balanced race distribution among the five races?")
for k, v in allowed_races.items():
    print(f"  {k} -> {v}")

use_balanced = prompt_with_default(
    "Enter 'yes' for balanced or 'no' to specify custom percentages",
    default="yes",
    caster=lambda s: s.lower()
)
use_balanced = "yes" if use_balanced not in ["yes", "no"] else use_balanced

# === Step 3: Get race percentages ===
if use_balanced == 'yes':
    race_dist = {k: 1.0 / len(allowed_races) for k in allowed_races}
else:
    print("\nPlease enter race percentages (0 - 100). Press Enter to accept default 20% each.")
    race_raw = {}
    for label in allowed_races:
        percent = prompt_with_default(
            f"How much percentage do you want for {allowed_races[label]}?",
            default=100.0 / len(allowed_races),
            caster=float,
        )
        if percent < 0:
            print("Negative value provided. Using 0.")
            percent = 0.0
        race_raw[label] = percent
    total = sum(race_raw.values())
    if abs(total - 100) > 1e-2 and total > 0:
        print("Percentages do not sum to 100. Normalizing automatically...")
    race_dist = {k: (v / total if total > 0 else 1.0 / len(allowed_races)) for k, v in race_raw.items()}

# === Step 4: Get gender percentages ===
print("\nPlease enter gender percentages (0 - 100). Press Enter to accept default 50/50.")
gender_raw = {}
for gender in allowed_genders:
    percent = prompt_with_default(
        f"How much percentage do you want for {gender}?",
        default=100.0 / len(allowed_genders),
        caster=float,
    )
    if percent < 0:
        print("Negative value provided. Using 0.")
        percent = 0.0
    gender_raw[gender] = percent
total = sum(gender_raw.values())
if abs(total - 100) > 1e-2 and total > 0:
    print("Percentages do not sum to 100. Normalizing automatically...")
gender_dist = {k: (v / total if total > 0 else 1.0 / len(allowed_genders)) for k, v in gender_raw.items()}

# === Step 5: Age range ===
while True:
    age_min = prompt_with_default("\nEnter minimum age", default=18, caster=int)
    age_max = prompt_with_default("Enter maximum age", default=65, caster=int)
    if age_min >= age_max:
        print("Minimum age must be less than maximum age.")
        continue
    break

# === Step 6: Number of participants ===
while True:
    num_participants = prompt_with_default("\nEnter number of participants to generate", default=42, caster=int)
    if num_participants <= 0:
        print("Number must be positive.")
        continue
    break

# === Step 7: Generate participants ===
participants = []
for _ in range(num_participants):
    participant_id = str(uuid.uuid4())
    race_key = random.choices(list(race_dist.keys()), weights=race_dist.values())[0]
    race = allowed_races[race_key]
    gender = random.choices(list(gender_dist.keys()), weights=gender_dist.values())[0]
    age = random.randint(age_min, age_max)

    participants.append({
        'ParticipantID': participant_id,
        'Race': race,
        'Gender': gender,
        'Age': age
    })

# === Step 8: Save to CSV ===
df = pd.DataFrame(participants)

# Get directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create full path for CSV file
output_path = os.path.join(script_dir, "participant_pool.csv")

# Save the file
df.to_csv(output_path, index=False)

print(f"\nParticipant pool saved to: {output_path}")
