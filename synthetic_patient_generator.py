import polars as pl
from faker import Faker
import random
import concurrent.futures
import sys
from datetime import datetime, timedelta
import uuid

# Constants for data generation
GENDERS = ["male", "female", "other"]
RACES = ["White", "Black", "Asian", "Hispanic", "Native American", "Other"]
ETHNICITIES = ["Not Hispanic or Latino", "Hispanic or Latino"]
MARITAL_STATUSES = ["Never Married", "Married", "Divorced", "Widowed", "Separated"]
LANGUAGES = ["English", "Spanish", "Chinese", "French", "German", "Vietnamese"]
INSURANCES = ["Medicare", "Medicaid", "Private", "Uninsured"]
ENCOUNTER_TYPES = ["Wellness Visit", "Emergency", "Follow-up", "Specialist", "Lab", "Surgery"]
ENCOUNTER_REASONS = ["Checkup", "Injury", "Illness", "Chronic Disease", "Vaccination", "Lab Work"]
CONDITION_NAMES = ["Hypertension", "Diabetes", "Asthma", "COPD", "Heart Disease", "Obesity", "Depression", "Anxiety", "Arthritis", "Cancer", "Flu", "COVID-19", "Migraine", "Allergy"]
CONDITION_STATUSES = ["active", "resolved", "remission"]
MEDICATIONS = ["Metformin", "Lisinopril", "Atorvastatin", "Albuterol", "Insulin", "Ibuprofen", "Amoxicillin", "Levothyroxine", "Amlodipine", "Omeprazole"]
ALLERGY_SUBSTANCES = ["Penicillin", "Peanuts", "Shellfish", "Latex", "Bee venom", "Aspirin", "Eggs", "Milk"]
ALLERGY_REACTIONS = ["Rash", "Anaphylaxis", "Hives", "Swelling", "Nausea", "Vomiting"]
ALLERGY_SEVERITIES = ["mild", "moderate", "severe"]
PROCEDURES = ["Appendectomy", "Colonoscopy", "MRI Scan", "X-ray", "Blood Test", "Vaccination", "Physical Therapy", "Cataract Surgery"]
IMMUNIZATIONS = ["Influenza", "COVID-19", "Tetanus", "Hepatitis B", "MMR", "Varicella", "HPV"]
OBSERVATION_TYPES = ["Height", "Weight", "Blood Pressure", "Heart Rate", "Temperature", "Hemoglobin A1c", "Cholesterol"]

fake = Faker()

def generate_patient(_):
    birthdate = fake.date_of_birth(minimum_age=0, maximum_age=100)
    age = (datetime.now().date() - birthdate).days // 365
    patient_id = str(uuid.uuid4())
    return {
        "patient_id": patient_id,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "gender": random.choice(GENDERS),
        "birthdate": birthdate.isoformat(),
        "age": age,
        "race": random.choice(RACES),
        "ethnicity": random.choice(ETHNICITIES),
        "address": fake.street_address(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip": fake.zipcode(),
        "country": "US",
        "marital_status": random.choice(MARITAL_STATUSES),
        "language": random.choice(LANGUAGES),
        "insurance": random.choice(INSURANCES),
        "ssn": fake.ssn(),
    }

def generate_encounters(patient, min_enc=1, max_enc=8):
    n = random.randint(min_enc, max_enc)
    encounters = []
    start_date = datetime.strptime(patient["birthdate"], "%Y-%m-%d")
    for _ in range(n):
        days_offset = random.randint(0, (datetime.now() - start_date).days)
        encounter_date = start_date + timedelta(days=days_offset)
        encounters.append({
            "encounter_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "date": encounter_date.date().isoformat(),
            "type": random.choice(ENCOUNTER_TYPES),
            "reason": random.choice(ENCOUNTER_REASONS),
            "provider": fake.company(),
            "location": fake.city(),
        })
    return encounters

def generate_conditions(patient, encounters, min_cond=1, max_cond=5):
    n = random.randint(min_cond, max_cond)
    conditions = []
    for _ in range(n):
        enc = random.choice(encounters) if encounters else None
        onset_date = enc["date"] if enc else patient["birthdate"]
        conditions.append({
            "condition_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"] if enc else None,
            "name": random.choice(CONDITION_NAMES),
            "status": random.choice(CONDITION_STATUSES),
            "onset_date": onset_date,
        })
    return conditions

def generate_medications(patient, encounters, min_med=0, max_med=4):
    n = random.randint(min_med, max_med)
    medications = []
    for _ in range(n):
        enc = random.choice(encounters) if encounters else None
        start_date = enc["date"] if enc else patient["birthdate"]
        # Convert start_date to datetime.date if needed
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            start_date_obj = start_date
        if random.random() < 0.7:
            end_date = None
        else:
            end_date = fake.date_between(start_date=start_date_obj, end_date="today").isoformat()
        medications.append({
            "medication_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"] if enc else None,
            "name": random.choice(MEDICATIONS),
            "start_date": start_date,
            "end_date": end_date,
        })
    return medications

def generate_allergies(patient, min_all=0, max_all=2):
    n = random.randint(min_all, max_all)
    allergies = []
    for _ in range(n):
        allergies.append({
            "allergy_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "substance": random.choice(ALLERGY_SUBSTANCES),
            "reaction": random.choice(ALLERGY_REACTIONS),
            "severity": random.choice(ALLERGY_SEVERITIES),
        })
    return allergies

def generate_procedures(patient, encounters, min_proc=0, max_proc=3):
    n = random.randint(min_proc, max_proc)
    procedures = []
    for _ in range(n):
        enc = random.choice(encounters) if encounters else None
        date = enc["date"] if enc else patient["birthdate"]
        procedures.append({
            "procedure_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"] if enc else None,
            "name": random.choice(PROCEDURES),
            "date": date,
            "outcome": random.choice(["successful", "complication", "failed"]),
        })
    return procedures

def generate_immunizations(patient, encounters, min_imm=0, max_imm=3):
    n = random.randint(min_imm, max_imm)
    immunizations = []
    for _ in range(n):
        enc = random.choice(encounters) if encounters else None
        date = enc["date"] if enc else patient["birthdate"]
        immunizations.append({
            "immunization_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"] if enc else None,
            "vaccine": random.choice(IMMUNIZATIONS),
            "date": date,
        })
    return immunizations

def generate_observations(patient, encounters, min_obs=1, max_obs=8):
    n = random.randint(min_obs, max_obs)
    observations = []
    for _ in range(n):
        enc = random.choice(encounters) if encounters else None
        date = enc["date"] if enc else patient["birthdate"]
        obs_type = random.choice(OBSERVATION_TYPES)
        value = None
        if obs_type == "Height":
            value = round(random.uniform(140, 200), 1)  # cm
        elif obs_type == "Weight":
            value = round(random.uniform(40, 150), 1)  # kg
        elif obs_type == "Blood Pressure":
            value = f"{random.randint(90, 180)}/{random.randint(60, 110)}"
        elif obs_type == "Heart Rate":
            value = random.randint(50, 120)
        elif obs_type == "Temperature":
            value = round(random.uniform(36.0, 39.0), 1)
        elif obs_type == "Hemoglobin A1c":
            value = round(random.uniform(4.5, 12.0), 1)
        elif obs_type == "Cholesterol":
            value = random.randint(120, 300)
        observations.append({
            "observation_id": str(uuid.uuid4()),
            "patient_id": patient["patient_id"],
            "encounter_id": enc["encounter_id"] if enc else None,
            "type": obs_type,
            "value": value,
            "date": date,
        })
    return observations

def main():
    try:
        n = int(input("How many synthetic patient records do you want to generate? "))
    except Exception:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)

    print(f"Generating {n} patients and related tables in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        patients = list(executor.map(generate_patient, range(n)))

    all_encounters = []
    all_conditions = []
    all_medications = []
    all_allergies = []
    all_procedures = []
    all_immunizations = []
    all_observations = []

    for patient in patients:
        encounters = generate_encounters(patient)
        all_encounters.extend(encounters)
        all_conditions.extend(generate_conditions(patient, encounters))
        all_medications.extend(generate_medications(patient, encounters))
        all_allergies.extend(generate_allergies(patient))
        all_procedures.extend(generate_procedures(patient, encounters))
        all_immunizations.extend(generate_immunizations(patient, encounters))
        all_observations.extend(generate_observations(patient, encounters))

    # Output each table as CSV and Parquet
    pl.DataFrame(patients).write_csv("patients.csv")
    pl.DataFrame(patients).write_parquet("patients.parquet")
    pl.DataFrame(all_encounters).write_csv("encounters.csv")
    pl.DataFrame(all_encounters).write_parquet("encounters.parquet")
    pl.DataFrame(all_conditions).write_csv("conditions.csv")
    pl.DataFrame(all_conditions).write_parquet("conditions.parquet")
    pl.DataFrame(all_medications).write_csv("medications.csv")
    pl.DataFrame(all_medications).write_parquet("medications.parquet")
    pl.DataFrame(all_allergies).write_csv("allergies.csv")
    pl.DataFrame(all_allergies).write_parquet("allergies.parquet")
    pl.DataFrame(all_procedures).write_csv("procedures.csv")
    pl.DataFrame(all_procedures).write_parquet("procedures.parquet")
    pl.DataFrame(all_immunizations).write_csv("immunizations.csv")
    pl.DataFrame(all_immunizations).write_parquet("immunizations.parquet")
    pl.DataFrame(all_observations).write_csv("observations.csv")
    pl.DataFrame(all_observations).write_parquet("observations.parquet")

    print("Done! Files written: patients, encounters, conditions, medications, allergies, procedures, immunizations, observations (CSV and Parquet)")

if __name__ == "__main__":
    main() 