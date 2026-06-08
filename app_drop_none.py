import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy import sparse


# Page setup
st.set_page_config(
    page_title="Student Depression Risk Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)


# Simple light styling for the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #111827;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
        color: #111827 !important;
    }

    input {
        color: #111827 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        color: #111827 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background-color: #1f77b4 !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
        box-shadow: none !important;
    }

    div[data-testid="stAlert"] {
        color: #111827 !important;
    }

    .stButton > button {
        background-color: #1f77b4;
        color: #ffffff;
        border: 0;
        border-radius: 6px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #155f91;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model_files():
    final_model = joblib.load("student_depression_xgboost_model_drop_none.pkl")
    feature_list = joblib.load("model_features_drop_none.pkl")
    preprocessor = final_model.named_steps["preprocessor"]
    classifier = final_model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    
    return final_model, feature_list, preprocessor, classifier, explainer


model, model_features, preprocessor, classifier, explainer = load_model_files()


def map_transformed_feature_name(transformed_name):
    clean_name = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    
    for original_feature in model_features:
        if clean_name == original_feature or clean_name.startswith(original_feature + "_"):
            return original_feature
    
    return clean_name


def format_feature_name(feature_name):
    display_names = {
        "Suicidal_Thoughts": "Suicidal Thoughts",
        "Work/Study Hours": "Study Hours per Day",
        "Family History of Mental Illness": "Family History",
        "Academic Pressure": "Academic Pressure",
        "Study Satisfaction": "Study Satisfaction",
        "Financial Stress": "Financial Stress",
        "Dietary Habits": "Dietary Habits",
        "Sleep Duration": "Sleep Duration",
        "CGPA": "CGPA",
        "Degree": "Degree",
        "Age": "Age",
        "Gender": "Gender"
    }
    
    return display_names.get(feature_name, feature_name)


def format_feature_value(value):
    if pd.isna(value):
        return "Not available"
    
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        
        return f"{float(value):.2f}"
    
    return str(value)


def get_local_shap_table(input_data, top_n=8):
    transformed_input = preprocessor.transform(input_data)
    
    if sparse.issparse(transformed_input):
        transformed_input = transformed_input.toarray()
    
    shap_values = explainer.shap_values(transformed_input)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    transformed_feature_names = preprocessor.get_feature_names_out()
    
    feature_map = {
        name: map_transformed_feature_name(name)
        for name in transformed_feature_names
    }
    
    shap_series = pd.Series(shap_values[0], index=transformed_feature_names)
    transformed_series = pd.Series(transformed_input[0], index=transformed_feature_names)
    rows = []
    
    for transformed_name in transformed_feature_names:
        clean_name = (
            transformed_name.split("__", 1)[1]
            if "__" in transformed_name
            else transformed_name
        )
        original_feature = feature_map[transformed_name]
        is_one_hot_category = clean_name != original_feature
        
        if is_one_hot_category and not np.isclose(transformed_series[transformed_name], 1):
            continue
        
        rows.append({
            "Feature": original_feature,
            "SHAP Contribution": shap_series[transformed_name],
            "Selected Value": (
                input_data.iloc[0][original_feature]
                if original_feature in input_data.columns
                else np.nan
            )
        })
    
    shap_table = pd.DataFrame(rows)
    shap_table["Absolute Contribution"] = shap_table["SHAP Contribution"].abs()
    shap_table["Feature Display"] = shap_table.apply(
        lambda row: (
            f"{format_feature_name(row['Feature'])} = "
            f"{format_feature_value(row['Selected Value'])}"
        ),
        axis=1
    )
    
    shap_table = (
        shap_table
        .sort_values(by="Absolute Contribution", ascending=False)
        .head(top_n)
        .drop(columns="Absolute Contribution")
    )
    
    return shap_table


def plot_shap_table(shap_table):
    plot_df = shap_table.sort_values("SHAP Contribution")
    colors = [
        "#d62728" if value > 0 else "#1f77b4"
        for value in plot_df["SHAP Contribution"]
    ]
    
    fig, ax = plt.subplots(figsize=(7, 4.8))
    
    ax.barh(
        plot_df["Feature Display"],
        plot_df["SHAP Contribution"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )
    
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution toward depression risk")
    ax.set_ylabel("")
    ax.set_title("Local SHAP Explanation Using Active Categories", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    fig.tight_layout()
    
    return fig


st.title("Student Depression Risk Prediction")

st.write(
    "This app estimates the probability of depression risk based on student-related information."
)

left_col, right_col = st.columns([1, 1.15], gap="large")


with left_col:
    st.subheader("Student Information")
    
    with st.form("prediction_form"):
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=59,
            value=21
        )
        
        academic_pressure = st.slider(
            "Academic Pressure",
            min_value=0,
            max_value=5,
            value=3
        )
        
        cgpa_not_available = st.checkbox("CGPA not available")
        
        if cgpa_not_available:
            cgpa = np.nan
        else:
            cgpa = st.number_input(
                "CGPA",
                min_value=5.0,
                max_value=10.0,
                value=7.00,
                step=0.01
            )
        
        study_satisfaction = st.slider(
            "Study Satisfaction",
            min_value=0,
            max_value=5,
            value=3
        )
        
        sleep_duration = st.selectbox(
            "Sleep Duration",
            [
                "Less than 5 hours",
                "5-6 hours",
                "7-8 hours",
                "More than 8 hours",
                "Others"
            ]
        )
        
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy", "Others"]
        )
        
        degree = st.selectbox(
            "Degree",
            [
                "Class 12",
                "B.Arch", "B.Com", "B.Ed", "B.Pharm", "B.Tech",
                "BA", "BBA", "BCA", "BE", "BHM", "BSc",
                "LLB", "LLM",
                "M.Com", "M.Ed", "M.Pharm", "M.Tech",
                "MA", "MBA", "MBBS", "MCA", "MD", "ME", "MHM", "MSc",
                "PhD",
                "Others"
            ]
        )
        
        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts?",
            ["No", "Yes"]
        )
        
        work_study_hours = st.slider(
            "Study Hours per Day",
            min_value=0,
            max_value=12,
            value=6
        )
        
        financial_stress = st.slider(
            "Financial Stress",
            min_value=1,
            max_value=5,
            value=3
        )
        
        family_history = st.selectbox(
            "Family History of Mental Illness",
            ["No", "Yes"]
        )
        
        submitted = st.form_submit_button(
        "Predict Depression Risk",
        type="secondary",
        use_container_width=False
)


with right_col:
    st.subheader("Prediction and Explanation")
    
    if submitted:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Academic Pressure": academic_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Degree": degree,
            "Suicidal_Thoughts": suicidal_thoughts,
            "Work/Study Hours": work_study_hours,
            "Financial Stress": financial_stress,
            "Family History of Mental Illness": family_history
        }])
        
        # Make sure column order matches training
        input_data = input_data[model_features]
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[:, 1][0]
        
        st.metric(
            label="Probability of Depression Risk",
            value=f"{probability:.3f}"
        )
        
        if prediction == 1:
            st.error("Prediction: At Risk of Depression")
        else:
            st.success("Prediction: Not At Risk of Depression")
        
        shap_table = get_local_shap_table(input_data, top_n=8)
        shap_figure = plot_shap_table(shap_table)
        st.pyplot(shap_figure)
        
        with st.expander("View SHAP values"):
            st.dataframe(
                shap_table[
                    [
                        "Feature Display",
                        "SHAP Contribution"
                    ]
                ].rename(columns={
                    "Feature Display": "Feature and Selected Value"
                }),
                use_container_width=True,
                hide_index=True
            )
        
        st.caption(
            "Red bars increase depression risk, while blue bars decrease depression risk. "
            "This prediction is generated by a machine learning model and should not be treated as a clinical diagnosis."
        )
    else:
        st.info("Enter the student information on the left, then click Predict Depression Risk.")
