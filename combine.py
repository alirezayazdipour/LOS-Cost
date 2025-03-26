import streamlit as st
import pandas as pd
import joblib

# Load models
los_model = joblib.load("xgb_los_model.pkl")
insurance_model = joblib.load("xgb_Insurance_cost_model.pkl")
patient_model = joblib.load("xgb_Patient_cost_model.pkl")

st.set_page_config(page_title="Hospital Prediction App", layout="centered")

# --- Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #2E86C1;
            text-align: center;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            padding: 8px 16px;
            font-size: 16px;
        }
        .card {
            margin-top: 25px;
            padding: 10px;
            border-radius: 10px;
            background-color: #ecfdf5;
            color: #065f46;
            font-weight: bold;
            text-align: center;
            font-size: 19px;
            border: 2px solid #10b981;
        }
        .cost-card {
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 80px;
        }
        .blue-card {
            background-color: #eaf3fc;
            color: #154360;
            border: 2px solid #1e67ba;
            font-size: 18px;
        }
        .red-card {
            background-color: #fdecea;
            color: #78281f;
            border: 2px solid #a83225;
            font-size: 18px;
        }
        .green-card {
            background-color: #eafaf1;
            color: #0b5345;
            border: 2px solid #117a65;
            font-size: 19px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(""" 
# ⚠️ What would you like to predict?
""")

prediction_type = st.selectbox("Select Prediction Type:", ["Select ...","🛌 Prediction of Hospital Length of Stay", "💵 Prediction of Patient, Insurance, and Total Hospital Costs"])

# Common Inputs
def common_inputs():
    st.markdown("""
        <h3> <span style='font-size: 25px;'>🧍‍♂️🧍‍♀️</span> Patient Information </h3>
        """, unsafe_allow_html=True)

    dem_col1, dem_col2 = st.columns(2)
    with dem_col1:
        age = st.number_input("Age", min_value=23, max_value=94, value=50)
    with dem_col2:
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"])

    st.markdown("""
        <h3> <span style='font-size: 25px;'>🩺</span> Intervention </h3>
        """, unsafe_allow_html=True)

    int_col1, int_col2 = st.columns(2)
    with int_col1:
        intervention = st.selectbox("Type of Intervention", ["Select...", "Angioplasty", "CABG"])
        angioplasty = 1 if intervention == "Angioplasty" else 0

    with int_col2:
        cabg_type = None
        if intervention == "CABG":
            cabg_type = st.selectbox("Type of CABG", [
                "Select CABG type...", "CABG (One Artery)", "CABG (Two Arteries)",
                "CABG (One Vein)", "CABG (Two or More Veins)"
            ])

    
    st.markdown("""
        <h3> <span style='font-size: 25px;'>🧬</span> Comorbidities </h3>
        """, unsafe_allow_html=True)

    com_col1, com_col2 = st.columns([1, 2])
    with com_col1:
        comorbidity = st.selectbox("Comorbidities", ["Select...", "Yes", "No"])

    meta = neuro = cardio = resp = kidney = "No"
    selected_diseases = []
    with com_col2:
        if comorbidity == "Yes":
            selected_diseases = st.multiselect("Type of Comorbidity", [
                "Metabolic and Endocrine Diseases", "Neurological and Brain Diseases",
                "Cardiovascular Diseases", "Respiratory Diseases", "Kidney Diseases"])
            meta = "Yes" if "Metabolic and Endocrine Diseases" in selected_diseases else "No"
            neuro = "Yes" if "Neurological and Brain Diseases" in selected_diseases else "No"
            cardio = "Yes" if "Cardiovascular Diseases" in selected_diseases else "No"
            resp = "Yes" if "Respiratory Diseases" in selected_diseases else "No"
            kidney = "Yes" if "Kidney Diseases" in selected_diseases else "No"

    return age, gender, intervention, angioplasty, cabg_type, comorbidity, meta, neuro, cardio, resp, kidney, selected_diseases

# --- LOS Prediction ---
if prediction_type == "🛌 Prediction of Hospital Length of Stay":
    age, gender, intervention, angioplasty, cabg_type, comorbidity, meta, neuro, cardio, resp, kidney, selected_diseases = common_inputs()

    # Validation
    missing = (
        "Select..." in [gender, intervention, comorbidity] or
        (intervention == "CABG" and (cabg_type is None or "Select" in cabg_type)) or
        (comorbidity == "Yes" and len(selected_diseases) == 0)
    )

    if st.button("Predict Length of Stay"):
        if missing:
            st.error("⚠️ Please complete all fields correctly.")
        else:
            input_data = {
                "Age": age,
                "Gender": 1 if gender == "Male" else 0,
                "Angioplasty": angioplasty,
                "CABG (One Artery)": 1 if cabg_type == "CABG (One Artery)" else 0,
                "CABG (Two Arteries)": 1 if cabg_type == "CABG (Two Arteries)" else 0,
                "CABG (One Vein)": 1 if cabg_type == "CABG (One Vein)" else 0,
                "CABG (Two or More Veins)": 1 if cabg_type == "CABG (Two or More Veins)" else 0,
                "Comorbidities (Yes, No)": 1 if comorbidity == "Yes" else 0,
                "Metabolic and Endocrine Diseases": 1 if meta == "Yes" else 0,
                "Neurological and Brain Diseases": 1 if neuro == "Yes" else 0,
                "Cardiovascular Diseases": 1 if cardio == "Yes" else 0,
                "Respiratory Diseases": 1 if resp == "Yes" else 0,
                "Kidney Diseases": 1 if kidney == "Yes" else 0
            }
            df = pd.DataFrame([input_data])
            los = los_model.predict(df)[0]
            if cabg_type == "CABG (One Artery)":
                los -= 2
            elif cabg_type in ["CABG (Two Arteries)", "CABG (One Vein)"]:
                los -= 1
            st.markdown(f"""
                <div class='card'>
                    Predicted Length of Stay<br><span style='font-size: 28px;'>{max(los, 0):.2f} Days</span>
                </div>
            """, unsafe_allow_html=True)

# --- Cost Prediction ---
elif prediction_type == "💵 Prediction of Patient, Insurance, and Total Hospital Costs":
    # Reuse shared inputs
    st.markdown("""
        <h3> <span style='font-size: 25px;'>🧍‍♂️🧍‍♀️</span> Patient Information </h3>
        """, unsafe_allow_html=True)

    dem_col1, dem_col2, dem_col3, dem_col4 = st.columns(4)
    with dem_col1:
        age = st.number_input("Age", min_value=23, max_value=94, value=50)
    with dem_col2:
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"])
    with dem_col3:
        insurance_type = st.selectbox("Insurance Type", ["Select...", "ArmedForces", "Free", "Private", "Veterans"], key="ins")
    with dem_col4:
        los = st.number_input("Length of Stay (days)", min_value=1, max_value=25, key="los_cost")

    # Intervention
    st.markdown("""
        <h3> <span style='font-size: 25px;'>🩺</span> Intervention </h3>
        """, unsafe_allow_html=True)

    int_col1, int_col2 = st.columns(2)
    with int_col1:
        intervention = st.selectbox("Type of Intervention", ["Select...", "Angioplasty", "CABG"])
        angioplasty = 1 if intervention == "Angioplasty" else 0
    with int_col2:
        cabg_type = None
        if intervention == "CABG":
            cabg_type = st.selectbox("Type of CABG", [
                "Select CABG type...", "CABG (One Artery)",
                "CABG (Two Arteries)", "CABG (One Vein)",
                "CABG (Two or More Veins)"
            ])

    # Comorbidities
    st.markdown("""
        <h3> <span style='font-size: 25px;'>🧬</span> Comorbidities </h3>
        """, unsafe_allow_html=True)

    com_col1, com_col2 = st.columns([1, 2])
    with com_col1:
        comorbidity = st.selectbox("Comorbidities", ["Select...", "Yes", "No"])
    meta = neuro = cardio = resp = kidney = "No"
    selected_diseases = []
    with com_col2:
        if comorbidity == "Yes":
            selected_diseases = st.multiselect("Type of Comorbidity", [
                "Metabolic and Endocrine Diseases", "Neurological and Brain Diseases",
                "Cardiovascular Diseases", "Respiratory Diseases", "Kidney Diseases"])
            meta = "Yes" if "Metabolic and Endocrine Diseases" in selected_diseases else "No"
            neuro = "Yes" if "Neurological and Brain Diseases" in selected_diseases else "No"
            cardio = "Yes" if "Cardiovascular Diseases" in selected_diseases else "No"
            resp = "Yes" if "Respiratory Diseases" in selected_diseases else "No"
            kidney = "Yes" if "Kidney Diseases" in selected_diseases else "No"

    # Validation
    missing = (
        "Select..." in [gender, intervention, insurance_type, comorbidity] or
        (intervention == "CABG" and (cabg_type is None or "Select" in cabg_type)) or
        (comorbidity == "Yes" and len(selected_diseases) == 0)
    )

    if st.button("Predict Hospital Costs"):
        if missing:
            st.error("⚠️ Please complete all fields correctly.")
        else:
            features = {
                "Age": age,
                "Gender": 1 if gender == "Male" else 0,
                "Insurance Type (ArmedForces)": 1 if insurance_type == "ArmedForces" else 0,
                "Insurance Type (Free)": 1 if insurance_type == "Free" else 0,
                "Insurance Type (Private)": 1 if insurance_type == "Private" else 0,
                "Insurance Type (Veterans)": 1 if insurance_type == "Veterans" else 0,
                "LOS": los,
                "Angioplasty": angioplasty,
                "CABG (One Artery)": 1 if cabg_type == "CABG (One Artery)" else 0,
                "CABG (Two Arteries)": 1 if cabg_type == "CABG (Two Arteries)" else 0,
                "CABG (One Vein)": 1 if cabg_type == "CABG (One Vein)" else 0,
                "CABG (Two or More Veins)": 1 if cabg_type == "CABG (Two or More Veins)" else 0,
                "Comorbidity": 1 if comorbidity == "Yes" else 0,
                "Metabolic and Endocrine Diseases": 1 if meta == "Yes" else 0,
                "Neurological and Brain Diseases": 1 if neuro == "Yes" else 0,
                "Cardiovascular Diseases": 1 if cardio == "Yes" else 0,
                "Respiratory Diseases": 1 if resp == "Yes" else 0,
                "Kidney Diseases": 1 if kidney == "Yes" else 0
            }

            df_input = pd.DataFrame([features])
            insurance_cost = 0 if insurance_type == "Free" else max(0, insurance_model.predict(df_input)[0])
            patient_cost = max(0, patient_model.predict(df_input)[0])
            total_cost = insurance_cost + patient_cost

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                    <div class='cost-card blue-card'>
                        Insurance Cost<br><span style='font-size: 25px;'>{insurance_cost:,.0f} USD</span>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class='cost-card red-card'>
                        Patient Cost<br><span style='font-size: 25px;'>{patient_cost:,.0f} USD</span>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class='cost-card green-card' style='margin-top: 15px;'>
                    Total Cost<br><span style='font-size: 28px;'>{total_cost:,.0f} USD</span>
                </div>
            """, unsafe_allow_html=True)
