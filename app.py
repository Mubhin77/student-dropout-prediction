
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.config import FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_dropout_model.joblib"

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# Get category options and numerical defaults
preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["categorical"]
scaler = preprocessor.named_transformers_["numerical"]

categorical_options = encoder.categories_
numerical_defaults = scaler.mean_

# --------------------------------------------------
# 3. Application heading
# --------------------------------------------------

st.title("🎓 Student Dropout Prediction System")

st.write(
    "Enter a student's admission information to estimate "
    "whether their eventual academic outcome is Graduate or Dropout."
)

st.info(
    "This application demonstrates a machine learning model "
    "trained on historical student records. It is not a validated "
    "early-warning or automatic decision-making system."
)

# --------------------------------------------------
# 4. Student information form
# --------------------------------------------------

st.subheader("Student Admission Information")

with st.form("student_prediction_form"):

    input_data = {}

    st.markdown("### Categorical Information")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(CATEGORICAL_FEATURES):

        options = categorical_options[index].tolist()

        with col1 if index % 2 == 0 else col2:
            input_data[feature] = st.selectbox(
                feature,
                options=options,
                key=f"categorical_{index}"
            )

    st.markdown("### Numerical Information")

    col3, col4 = st.columns(2)

    for index, feature in enumerate(NUMERICAL_FEATURES):

        default_value = float(numerical_defaults[index])

        with col3 if index % 2 == 0 else col4:
            input_data[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.3f",
                key=f"numerical_{index}"
            )

    submitted = st.form_submit_button(
        "Predict Student Outcome",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Make prediction
# --------------------------------------------------

if submitted:

    student_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    prediction = int(model.predict(student_df)[0])

    dropout_probability = float(
        model.predict_proba(student_df)[0][1]
    )

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Predicted Outcome: Dropout")
    else:
        st.success("Predicted Outcome: Graduate")

    st.metric(
        "Estimated Dropout Probability",
        f"{dropout_probability:.2%}"
    )

    st.progress(dropout_probability)

    st.caption(
        "The probability is a model estimate, not a certainty. "
        "The result should be interpreted alongside other relevant "
        "student information."
    )

    with st.expander("View submitted student information"):
        st.dataframe(student_df, use_container_width=True)
