
import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "student_dropout_model_12_features.joblib"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "student_dropout_model_12_features_config.json"
)

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# 2. LOAD TRAINED MODEL AND CONFIGURATION
# ==================================================

@st.cache_resource
def load_model_and_config():

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return model, config


try:

    model, model_config = load_model_and_config()

except Exception as error:

    st.error(
        f"Could not load the trained model or configuration: {error}"
    )

    st.stop()


# Get final model settings
FEATURES = model_config["selected_features"]

DECISION_THRESHOLD = float(
    model_config["decision_threshold"]
)


# ==================================================
# 3. EXTRACT PREPROCESSING INFORMATION
# ==================================================

preprocessor = model.named_steps["preprocessor"]

# Numerical and categorical feature names
numerical_features = list(
    preprocessor.named_transformers_["num"].feature_names_in_
)

categorical_features = list(
    preprocessor.named_transformers_["cat"].feature_names_in_
)

# Get encoder and scaler
encoder = preprocessor.named_transformers_["cat"]

scaler = preprocessor.named_transformers_["num"]

categorical_options = encoder.categories_

numerical_defaults = scaler.mean_


# ==================================================
# 4. HUMAN-READABLE CATEGORY LABELS
# ==================================================

CATEGORY_LABELS = {

    # Gender
    "Gender": {
        0: "Female",
        1: "Male"
    },

    # Binary features
    "Tuition fees up to date": {
        0: "No - fees are not up to date",
        1: "Yes - fees are up to date"
    },

    "Debtor": {
        0: "No - student is not a debtor",
        1: "Yes - student is a debtor"
    },

    "Scholarship holder": {
        0: "No - not a scholarship holder",
        1: "Yes - scholarship holder"
    },

    # Course
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (Evening)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (Evening)"
    },

    # Application mode
    "Application mode": {
        1: "1st phase - General contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - Special contingent (Azores)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (Bachelor)",
        16: "1st phase - Special contingent (Madeira)",
        17: "2nd phase - General contingent",
        18: "3rd phase - General contingent",
        26: "Ordinance No. 533-A/99 (Different Plan)",
        27: "Ordinance No. 533-A/99 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    },

    # Mother's occupation
    "Mother's occupation": {
        0: "Student",
        1: "Legislative/executive leaders and managers",
        2: "Intellectual and scientific professionals",
        3: "Intermediate-level technicians",
        4: "Administrative staff",
        5: "Personal services, security and sales workers",
        6: "Agriculture, fisheries and forestry workers",
        7: "Industry, construction and craft workers",
        8: "Machine operators and assemblers",
        9: "Unskilled workers",
        10: "Armed forces professions",
        90: "Other situation",
        99: "Not specified",
        122: "Health professionals",
        123: "Teachers",
        125: "ICT professionals",
        131: "Science and engineering technicians",
        132: "Health technicians",
        134: "Legal, social, sports and cultural technicians",
        141: "Office workers and secretaries",
        143: "Accounting and financial service operators",
        144: "Administrative support staff",
        151: "Personal service workers",
        152: "Sales workers",
        153: "Personal care workers",
        171: "Skilled construction workers",
        173: "Printing and precision craft workers",
        175: "Food, woodworking and clothing industry workers",
        191: "Cleaning workers",
        192: "Unskilled agricultural workers",
        193: "Unskilled industrial and construction workers",
        194: "Meal preparation assistants"
    },

    # Father's occupation
    "Father's occupation": {
        0: "Student",
        1: "Legislative/executive leaders and managers",
        2: "Intellectual and scientific professionals",
        3: "Intermediate-level technicians",
        4: "Administrative staff",
        5: "Personal services, security and sales workers",
        6: "Agriculture, fisheries and forestry workers",
        7: "Industry, construction and craft workers",
        8: "Machine operators and assemblers",
        9: "Unskilled workers",
        10: "Armed forces professions",
        90: "Other situation",
        99: "Not specified",
        101: "Armed forces officers",
        102: "Armed forces sergeants",
        103: "Other armed forces personnel",
        112: "Administrative and commercial service directors",
        114: "Hotel, catering and trade directors",
        121: "Physical science, mathematics and engineering professionals",
        122: "Health professionals",
        123: "Teachers",
        124: "Finance, accounting and administration professionals",
        131: "Science and engineering technicians",
        132: "Health technicians",
        134: "Legal, social, sports and cultural technicians",
        135: "ICT technicians",
        141: "Office workers and secretaries",
        143: "Accounting and financial service operators",
        144: "Administrative support staff",
        151: "Personal service workers",
        152: "Sales workers",
        153: "Personal care workers",
        154: "Protection and security personnel",
        161: "Market-oriented farmers",
        163: "Subsistence farmers, livestock keepers and fishers",
        171: "Skilled construction workers",
        172: "Metalworking workers",
        174: "Electrical and electronics workers",
        175: "Food, woodworking and clothing industry workers",
        181: "Fixed plant and machine operators",
        182: "Assembly workers",
        183: "Vehicle drivers and mobile equipment operators",
        192: "Unskilled agricultural workers",
        193: "Unskilled industrial and construction workers",
        194: "Meal preparation assistants",
        195: "Street vendors and street service providers"
    }
}


def format_category(feature, value):

    code = int(value)

    label = CATEGORY_LABELS.get(feature, {}).get(
        code,
        f"Other / Code {code}"
    )

    return label


# ==================================================
# 5. APPLICATION HEADING
# ==================================================

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

st.caption(
    f"Model: {model_config['model_name']} | "
    f"Features: {model_config['number_of_features']} | "
    f"Dropout decision threshold: {DECISION_THRESHOLD:.0%}"
)


# ==================================================
# 6. STUDENT INFORMATION FORM
# ==================================================

st.subheader("Student Admission Information")

with st.form("student_prediction_form"):

    input_data = {}

    # ----------------------------------------------
    # Categorical information
    # ----------------------------------------------

    st.markdown("### Categorical Information")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(categorical_features):

        options = categorical_options[index].tolist()

        with col1 if index % 2 == 0 else col2:

            input_data[feature] = st.selectbox(
                feature,
                options=options,
                format_func=lambda value, f=feature: format_category(
                    f, value
                ),
                key=f"categorical_{index}"
            )

    # ----------------------------------------------
    # Numerical information
    # ----------------------------------------------

    st.markdown("### Numerical Information")

    col3, col4 = st.columns(2)

    GRADE_FEATURES = [
        "Previous qualification (grade)",
        "Admission grade"
    ]

    GRADE_HELP = {

        "Previous qualification (grade)": (
            "Grade obtained in the student's previous qualification. "
            "Original dataset grading scale: 0 to 200."
        ),

        "Admission grade": (
            "Grade obtained during the student's admission process. "
            "Original dataset grading scale: 0 to 200."
        )
    }

    for index, feature in enumerate(numerical_features):

        default_value = float(numerical_defaults[index])

        with col3 if index % 2 == 0 else col4:

            if feature in GRADE_FEATURES:

                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=200.0,
                    value=default_value,
                    step=0.1,
                    format="%.1f",
                    help=GRADE_HELP[feature],
                    key=f"numerical_{index}"
                )

            elif feature == "Age at enrollment":

                input_data[feature] = st.number_input(
                    feature,
                    min_value=15,
                    max_value=100,
                    value=int(round(default_value)),
                    step=1,
                    help="Student's age at the time of enrollment.",
                    key=f"numerical_{index}"
                )

            else:

                input_data[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.3f",
                    help=(
                        "Enter the value recorded at student admission."
                    ),
                    key=f"numerical_{index}"
                )

    # ----------------------------------------------
    # Submit form
    # ----------------------------------------------

    submitted = st.form_submit_button(
        "Predict Student Outcome",
        use_container_width=True
    )


# ==================================================
# 7. MAKE PREDICTION
# ==================================================

if submitted:

    # Arrange inputs in exact model feature order
    student_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Calculate dropout probability
    dropout_probability = float(
        model.predict_proba(student_df)[0, 1]
    )

    # Apply final decision threshold
    prediction = int(
        dropout_probability >= DECISION_THRESHOLD
    )

    # Calculate graduate probability
    graduate_probability = 1 - dropout_probability

    st.divider()

    st.subheader("Prediction Result")

    # ----------------------------------------------
    # Prediction display
    # ----------------------------------------------

    if prediction == 1:

        st.error("### Predicted Outcome: Dropout")

    else:

        st.success("### Predicted Outcome: Graduate")

    # ----------------------------------------------
    # Probability metrics
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Dropout Probability",
            f"{dropout_probability:.2%}"
        )

    with col2:

        st.metric(
            "Estimated Graduate Probability",
            f"{graduate_probability:.2%}"
        )

    # ----------------------------------------------
    # Probability visualization
    # ----------------------------------------------

    probability_df = pd.DataFrame(
        {
            "Outcome": ["Dropout", "Graduate"],
            "Probability": [
                dropout_probability,
                graduate_probability
            ]
        }
    )

    st.markdown("### Outcome Probability")

    st.bar_chart(
        probability_df.set_index("Outcome")
    )

    # ----------------------------------------------
    # Decision explanation
    # ----------------------------------------------

    if prediction == 1:

        st.warning(
            f"The estimated dropout probability "
            f"({dropout_probability:.2%}) is at or above "
            f"the decision threshold of {DECISION_THRESHOLD:.0%}."
        )

    else:

        st.success(
            f"The estimated dropout probability "
            f"({dropout_probability:.2%}) is below "
            f"the decision threshold of {DECISION_THRESHOLD:.0%}."
        )

    st.caption(
        "The probability is a model estimate, not a certainty. "
        "Predictions should be interpreted alongside other relevant "
        "student information and should not be used as the sole basis "
        "for academic decisions."
    )

    # ----------------------------------------------
    # Submitted student information
    # ----------------------------------------------

    with st.expander("View submitted student information"):

        st.dataframe(
            student_df,
            use_container_width=True
        )


# ==================================================
# 8. MODEL INFORMATION
# ==================================================

with st.expander("About this machine learning model"):

    st.write(
        "This application uses a Logistic Regression model "
        "trained to distinguish between students who eventually "
        "Graduate and students who Dropout."
    )

    st.write("**Model configuration**")

    st.write(f"- Model: {model_config['model_name']}")
    st.write(f"- Number of features: {model_config['number_of_features']}")
    st.write(f"- Training records: {model_config['training_records']}")
    st.write(f"- Test records: {model_config['test_records']}")
    st.write(f"- Decision threshold: {DECISION_THRESHOLD:.2f}")

    st.write("**Held-out test performance**")

    metrics = model_config.get("metrics", {})

    metric_labels = {
        "accuracy": "Accuracy",
        "dropout_precision": "Dropout Precision",
        "dropout_recall": "Dropout Recall",
        "dropout_f1": "Dropout F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC"
    }

    for key, label in metric_labels.items():

        if key in metrics:

            st.write(
                f"- {label}: {metrics[key]:.4f}"
            )

    st.caption(
        "The model was trained using historical records from "
        "one institution. Its performance may differ on other "
        "student populations or institutions."
    )
