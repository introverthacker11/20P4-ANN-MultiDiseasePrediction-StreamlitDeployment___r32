import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -------------------- CACHE HELPERS --------------------
@st.cache_resource
def load_dl_model(model_path):
    return load_model(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    with open(scaler_path, "rb") as file:
        return pickle.load(file)

# -------------------- LOAD MODELS + SCALERS --------------------
heart_model = load_dl_model("HAP Model.h5")
heart_scaler = load_scaler("HAPM_StandardScaler.pkl")

diabetes_model = load_dl_model("diabetes_ann_model.keras")
diabetes_scaler = load_scaler("diabetes_scaler.pkl")

breast_model = load_dl_model("breast_cancer_ann_model.keras")
breast_scaler = load_scaler("brstcancer_scaler.pkl")

# -------------------- SIDEBAR --------------------

st.markdown("""
    <style>
    /* Sidebar custom style */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 50, 70, 0.6);  /* Dark blue-ish tone */
        color: white;
    }

    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }

    /* Optional: make sidebar title/headings colored */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00171F;  /* Light cyan */
    }

    /* Optional: control scrollbar style inside sidebar */
    ::-webkit-scrollbar-thumb {
        background: #00cfff;
        border-radius: 10px;
    }
    </style>
            
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 20px !important; /* increase size */
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🦠 Disease Prediction System")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

choice = st.sidebar.radio(
    "🦠 Select a Disease Model:",
    ["Heart Attack Prediction", "Diabetes Prediction", "Breast Cancer Prediction"]
)

with st.sidebar.expander("📁 Project Intro"):
    st.markdown("- **This is a Multi Disease Risk Prediction web app using an Artificial Neural Network (ANN)." \
    "It takes medical input features and predicts the likelihood of a Heart Attack, Diabetes, Breast Cancer.**")
 

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- **Numpy**")
    st.markdown("- **Pandas**")
    st.markdown("- **Matplotlib**")
    st.markdown("- **Seaborn**")
    st.markdown("- **Scikit Learn**")
    st.markdown("- **TensorFlow, Keras, Pickle**")
    st.markdown("- **Streamlit**")

# ================================================================= HEART ATTACK =================================================================

if choice == "Heart Attack Prediction":

    st.markdown("""
        <style>
        .stApp {
            background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://www.researchtrials.org/wp-content/uploads/2021/03/iStock-1128931450-scaled.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }

        h1 {
            color: #FFD700;  /* Gold */
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


    st.markdown(
        """
        <style>
        .glow-text {
            font-size: 50px;
            color: #ffffff;
            text-align: center;
            text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
            font-weight: bold;
        }
        </style>
        <div class="glow-text">💓 Heart Attack Risk Predictor</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='color: white; font-size: 20px; font-family: Arial; font-weight: bold'>📝 Enter your health information below:</h4>",
        unsafe_allow_html=True
    )

    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
                    help="cp type -> 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholestoral (chol)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = True, 0 = False")
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2], 
                        help="0: Normal, 1: ST-T abnormality, 2: Probable/definite left ventricular hypertrophy")
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], help="1 = Yes, 0 = No")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.2, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4], help="Colored by fluoroscopy")
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], help="1: Fixed Defect, 2: Normal, 3: Reversible Defect")

    if st.button("Predict"):
        
        try:
        
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
            
            scaled_features = heart_scaler.transform(features)
            
            prediction = heart_model.predict(scaled_features)[0][0]
            risk = prediction * 100
            
            st.subheader("🩺 Result:")
            if prediction > 0.5:
                st.error(f"⚠️ High Risk of Heart Attack ({risk:.3f}%)")
                st.snow()
                st.markdown(
                    """
                    <div style="background-color:#ff4d4d; padding:15px; border-radius:10px; color:white; font-size:18px; font-weight:bold;">
                        🚑 <b>Please consult a doctor immediately!</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            else:
                st.success(f"✅ Low Risk of Heart Attack ({risk:.3f}%)")
                st.balloons()
                st.markdown(
                    """
                    <div style="background-color:green; padding:15px; border-radius:10px; color:white; font-size:18px; font-weight:bold;">
                        🧘 <b>You're doing great! Keep maintaining a healthy lifestyle.</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error("❌ Something went wrong during prediction.")
            st.code(str(e))

# ======================================================================= DIABETES =======================================================================


elif choice == "Diabetes Prediction":

    st.markdown("""
        <style>
        .stApp {
            background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://hsmc.com.au/wp-content/uploads/2024/11/Diabetes.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }

        h1 {
            color: white;  /* Gold */
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .glow-text {
            font-size: 50px;
            color: #ffffff;
            text-align: center;
            text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
            font-weight: bold;
        }
        </style>
        <div class="glow-text">Diabetes Prediction App</div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Enter Patient Details:")

    # Feature inputs
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=0.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=140.0, value=0.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=0.0)
    Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=0.0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
    Age = st.number_input("Age", min_value=0, max_value=100, value=0)

    if st.button("Predict"):
        # Prepare and scale input
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_scaled = diabetes_scaler.transform(input_data)

        # Predict
        prediction = diabetes_model.predict(input_scaled)[0][0]

        # Output
        prob = prediction  # model output (0 to 1)

        percent = prob * 100

        if percent <= 5:
            st.success(f"Risk Level: Low ({percent:.2f}%)")
            st.balloons()

        elif percent <= 20:
            st.info(f"Risk Level: Mild ({percent:.2f}%)")
            st.snow()

        elif percent <= 40:
            st.warning(f"Risk Level: Moderate — needs lifestyle changes & monitoring ({percent:.2f}%)")
            st.snow()

        else:
            st.error(f"Risk Level: High — seek medical advice ({percent:.2f}%)")
            st.snow()

# -------------------- BREAST CANCER --------------------

elif choice == "Breast Cancer Prediction":

    st.markdown("""
        <style>
        .stApp {
            background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://media.istockphoto.com/id/1302922398/photo/top-view-of-bright-pink-ribbon-on-dark-wood-background-breast-cancer-awareness-and-womens.jpg?s=612x612&w=0&k=20&c=wpD3n37ipEmjV72utgh6SflUPvyzjDOu203_tlIbCMU=");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }

        h1 {
            color: #FFD700;  /* Gold */
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .glow-text {
            font-size: 50px;
            color: #ffffff;
            text-align: center;
            text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
            font-weight: bold;
        }
        </style>
        <div class="glow-text">🎗 Breast Cancer Risk Predictor</div>
        """,
        unsafe_allow_html=True
    )

    st.title("Breast Cancer Prediction using ANN 🤖🧠")
    st.write("Provide the required inputs below to predict whether the tumor is **Benign (0)** or **Malignant (1)**.")

    #st.title("Breast Cancer Prediction")
    feature_names = [
    'radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'texture_se', 'perimeter_se', 'radius_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst'
    ]

    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, format="%.3f")
        user_input.append(val)

    # Predict button
    if st.button("Predict"):
        input_array = np.array([user_input])
        scaled_input = breast_scaler.transform(input_array)
        prediction = breast_model.predict(scaled_input)
        output = (prediction > 0.5).astype(int)

        st.subheader("🔎 Prediction Result:")
        if output[0][0] == 1:
            st.error("The tumor is **Malignant (Cancerous)** ❌")
            st.snow()
        else:
            st.success("The tumor is **Benign (Non-cancerous)** ✅")
            st.balloons()

    
