
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import json

# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------
st.set_page_config(
    page_title="Motz Financial Services",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# CUSTOM DESIGN
# -----------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 0.8rem !important;
    max-width: 97%;
}

[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

[data-testid="stSidebar"] {
    background-color: #102747;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: white;
}

.main-header {
    background: linear-gradient(120deg, #102747, #24527e);
    padding: 12px 22px;
    border-radius: 10px;
    color: white;
    border-bottom: 3px solid #d5a52b;
    margin-bottom: 10px;
}

.main-header h1 {
    color: white;
    font-size: 23px;
    margin: 0 0 3px 0;
    padding: 0;
    letter-spacing: 1px;
}

.main-header p {
    color: white;
    font-size: 13px;
    margin: 2px 0;
    line-height: 1.3;
}

h1 {
    color: #102747 !important;
    font-size: 27px !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}

h2 {
    color: #102747 !important;
    font-size: 22px !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.3rem !important;
}

h3 {
    color: #24527e !important;
    font-size: 17px !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}

.metric-card {
    background: white;
    padding: 12px;
    border-radius: 9px;
    border-left: 4px solid #d5a52b;
    box-shadow: 0 2px 7px rgba(0,0,0,0.06);
}

.metric-card h3 {
    color: #24527e;
    font-size: 14px;
    margin: 0 0 5px 0 !important;
}

.metric-card h2 {
    color: #102747;
    font-size: 23px;
    margin: 0 !important;
}

[data-testid="stMetric"] {
    background: white;
    padding: 10px;
    border-radius: 8px;
    border-bottom: 2px solid #d5a52b;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #102747, #24527e);
    color: white;
    border: 1px solid #d5a52b;
    border-radius: 7px;
    font-weight: bold;
    padding: 8px 15px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #d5a52b;
    color: #102747;
    border-color: #102747;
}

[data-testid="stDataFrame"] {
    border-radius: 8px;
}

[data-testid="stExpander"] {
    background-color: white;
    border: 1px solid #dce4ef;
    border-radius: 8px;
}

.footer {
    text-align: center;
    color: white;
    background: #102747;
    padding: 9px;
    border-top: 3px solid #d5a52b;
    border-radius: 7px;
    font-size: 13px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD PROJECT FILES
# -----------------------------------------

@st.cache_data
def load_dataset():
    data = pd.read_csv("cleaned_loan_dataset.csv")
    data.columns = data.columns.str.strip().str.lower()
    return data


@st.cache_data
def load_raw_dataset():
    filename = "loan_approval_dataset.csv"

    if os.path.exists(filename):
        data = pd.read_csv(filename)
        data.columns = data.columns.str.strip().str.lower()
        return data

    return None


@st.cache_resource
def load_trained_model():
    return joblib.load("best_loan_model.joblib")


@st.cache_data
def load_comparison():
    if os.path.exists("model_comparison.csv"):
        return pd.read_csv("model_comparison.csv")
    return None


@st.cache_data
def load_tuning_information():
    if os.path.exists("model_tuning_results.json"):
        with open("model_tuning_results.json", "r") as file:
            return json.load(file)
    return None


@st.cache_data
def load_classification_report():
    if os.path.exists("classification_report.csv"):
        return pd.read_csv(
            "classification_report.csv",
            index_col=0
        )
    return None


# Load dataset
try:
    df = load_dataset()
    dataset_error = None
except Exception as error:
    df = None
    dataset_error = str(error)

raw_df = load_raw_dataset()

# Load model and preprocessing information
model = None
feature_columns = []
categorical_columns = []
numerical_columns = []
category_levels = {}
model_error = None

try:
    model_bundle = load_trained_model()

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    categorical_columns = model_bundle.get(
        "categorical_columns", []
    )

    numerical_columns = model_bundle.get(
        "numerical_columns", []
    )

    category_levels = model_bundle.get(
        "category_levels", {}
    )

except Exception as error:
    model_error = str(error)

# Load experiment results
comparison = load_comparison()
tuning_info = load_tuning_information()
classification_results = load_classification_report()


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def clean_status(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )


def get_number(data, key, default=0):
    if data is None:
        return default

    try:
        return int(data[key])
    except (KeyError, TypeError, ValueError):
        return default


def display_chart(fig):
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown("""
<div class="main-header">
    <h1>MOTZ FINANCIAL SERVICES</h1>
    <p><strong>LOAN PREDICTION SYSTEM</strong></p>
    <p>Machine Learning Based Loan Approval Platform</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------
# NAVIGATION
# -----------------------------------------

st.sidebar.title("MOTZ FINANCIAL SERVICES")
st.sidebar.write("Loan Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Prediction",
        "Data Analysis",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Loan Prediction Project")


# =========================================
# HOME PAGE
# =========================================

if page == "Home":

    st.title("Welcome to Motz Financial Services")

    st.write("""
    This system uses machine learning to predict whether a loan
    application is likely to be approved or rejected.

    It analyses applicant financial information to demonstrate
    how machine learning can support loan assessment.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - To gather and prepare loan application data.
    - To analyse applicant financial information.
    - To train and compare machine learning classification models.
    - To identify a suitable model for loan prediction.
    - To develop an interactive loan prediction dashboard.
    """)

    st.subheader("Project Overview")

    if df is not None and "loan_status" in df.columns:

        total = len(df)

        status = clean_status(df["loan_status"])

        approved = (status == "approved").sum()
        rejected = (status == "rejected").sum()

        approval_rate = (
            approved / total * 100 if total else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Applications</h3>
                <h2>{total:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Approved Loans</h3>
                <h2>{approved:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rejected Loans</h3>
                <h2>{rejected:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Approval Rate</h3>
                <h2>{approval_rate:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Dataset file could not be loaded.")
        if dataset_error:
            st.caption(dataset_error)

    st.subheader("Main System Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Prediction**\n\n"
            "Enter applicant details to receive a loan prediction."
        )

        st.info(
            "**Data Analysis**\n\n"
            "Explore the dataset, cleaning process and visualisations."
        )

    with col2:
        st.info(
            "**Model Performance**\n\n"
            "Compare machine learning models and view their results."
        )

        st.info(
            "**Home**\n\n"
            "View the project introduction and dataset overview."
        )

    st.subheader("How the System Works")

    st.markdown("""
    1. Loan application data is collected.
    2. The data is cleaned and prepared for machine learning.
    3. Different machine learning models are trained.
    4. The models are evaluated and compared.
    5. The selected model predicts the applicant's loan status.
    """)


# =========================================
# PREDICTION PAGE
# =========================================

elif page == "Prediction":

    st.title("Loan Prediction")

    st.write("Enter the applicant's information below.")

    if model is None:

        st.error(
            "The trained model could not be loaded. "
            "Check best_loan_model.joblib."
        )

        if model_error:
            st.code(model_error)

    else:

        st.subheader("Applicant Information")

        # Use the categories saved during model training.
        education_options = category_levels.get(
            "education",
            ["Graduate", "Not Graduate"]
        )

        employment_options = category_levels.get(
            "self_employed",
            ["No", "Yes"]
        )

        with st.form("loan_prediction_form"):

            col1, col2, col3 = st.columns(3)

            # PERSONAL INFORMATION
            with col1:

                st.markdown("### Personal Information")

                dependents = st.selectbox(
                    "Number of Dependents",
                    [0, 1, 2, 3, 4, 5]
                )

                education = st.selectbox(
                    "Education",
                    education_options
                )

                self_employed = st.selectbox(
                    "Self Employed",
                    employment_options
                )

                cibil_score = st.number_input(
                    "CIBIL Score",
                    min_value=300,
                    max_value=900,
                    value=700
                )

            # LOAN INFORMATION
            with col2:

                st.markdown("### Loan Information")

                income_annum = st.number_input(
                    "Annual Income",
                    min_value=0,
                    value=500000
                )

                loan_amount = st.number_input(
                    "Loan Amount",
                    min_value=0,
                    value=1000000
                )

                loan_term = st.number_input(
                    "Loan Term",
                    min_value=1,
                    value=10
                )

            # ASSET INFORMATION
            with col3:

                st.markdown("### Asset Information")

                residential_assets_value = st.number_input(
                    "Residential Assets Value",
                    min_value=0,
                    value=500000
                )

                commercial_assets_value = st.number_input(
                    "Commercial Assets Value",
                    min_value=0,
                    value=200000
                )

                luxury_assets_value = st.number_input(
                    "Luxury Assets Value",
                    min_value=0,
                    value=300000
                )

                bank_asset_value = st.number_input(
                    "Bank Asset Value",
                    min_value=0,
                    value=200000
                )

            submitted = st.form_submit_button(
                "Predict Loan Status",
                use_container_width=True
            )

        if submitted:

            applicant = pd.DataFrame([{
                "no_of_dependents": dependents,
                "education": str(education).strip(),
                "self_employed": str(self_employed).strip(),
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value
            }])

            try:

                # Ensure categorical values use the same categories
                # and ordering as the training dataset.
                for column in categorical_columns:

                    if column in applicant.columns:

                        levels = category_levels.get(column, [])

                        if levels:
                            applicant[column] = pd.Categorical(
                                applicant[column],
                                categories=levels
                            )

                # Encode categorical values
                applicant_encoded = pd.get_dummies(
                    applicant,
                    columns=categorical_columns,
                    drop_first=True,
                    dtype=int
                )

                # Match the exact training feature columns
                applicant_encoded = applicant_encoded.reindex(
                    columns=feature_columns,
                    fill_value=0
                )

                # Ensure numerical input
                applicant_encoded = applicant_encoded.astype(float)

                # Predict
                prediction = model.predict(
                    applicant_encoded
                )[0]

                predicted_status = (
                    str(prediction).strip().lower()
                )

                st.markdown("---")
                st.subheader("Prediction Result")

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        applicant_encoded
                    )[0]

                    classes = [
                        str(c).strip().lower()
                        for c in model.classes_
                    ]

                    if predicted_status == "approved":

                        probability = probabilities[
                            classes.index("approved")
                        ]

                        st.success(
                            "LOAN APPROVED (Model Prediction)"
                        )

                        st.metric(
                            "Approval Probability",
                            f"{probability * 100:.2f}%"
                        )

                    elif predicted_status == "rejected":

                        probability = probabilities[
                            classes.index("rejected")
                        ]

                        st.error(
                            "LOAN REJECTED (Model Prediction)"
                        )

                        st.metric(
                            "Rejection Probability",
                            f"{probability * 100:.2f}%"
                        )

                    else:

                        st.warning(
                            f"Predicted result: {prediction}"
                        )

                else:

                    st.write(
                        f"Predicted loan status: **{prediction}**"
                    )

                st.caption(
                    "This is a machine learning estimate, not a "
                    "guaranteed lending decision."
                )

            except Exception as error:

                st.error(
                    "The prediction could not be completed."
                )

                st.code(str(error))


# =========================================
# DATA ANALYSIS PAGE
# =========================================

elif page == "Data Analysis":

    st.title("Data Analysis")

    st.write("""
    This section explains how the loan dataset was gathered,
    examined and prepared for machine learning.
    """)

    if df is None:

        st.error("Dataset file could not be loaded.")

        if dataset_error:
            st.code(dataset_error)

    else:

        # ---------------------------------
        # 1. DATA GATHERING
        # ---------------------------------

        st.header("1. Data Gathering")

        st.write("""
        The project uses the loan approval dataset stored in
        loan_approval_dataset.csv. It contains applicant financial
        information and the loan status used as the prediction target.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Original Records",
                len(raw_df) if raw_df is not None else "Not available"
            )

        with col2:
            st.metric(
                "Original Columns",
                len(raw_df.columns) if raw_df is not None else "Not available"
            )

        with col3:
            st.metric(
                "Cleaned Records",
                len(df)
            )

        # ---------------------------------
        # 2. DATA CLEANING
        # ---------------------------------

        st.header("2. Data Cleaning")

        st.write("""
        The dataset was examined before model training. The checks
        included missing values, duplicate records, column names
        and the removal of the loan identifier from the model inputs.
        """)

        if raw_df is not None:

            raw_missing = int(
                raw_df.isnull().sum().sum()
            )

            raw_duplicates = int(
                raw_df.duplicated().sum()
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Missing Values in Original Data",
                    raw_missing
                )

            with col2:
                st.metric(
                    "Duplicate Records in Original Data",
                    raw_duplicates
                )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Missing Values in Cleaned Data",
                int(df.isnull().sum().sum())
            )

        with col2:
            st.metric(
                "Duplicate Records in Cleaned Data",
                int(df.duplicated().sum())
            )

        st.subheader("Missing Values by Column")

        missing_table = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing_table.columns = [
            "Column",
            "Missing Values"
        ]

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------
        # OUTLIER ANALYSIS
        # ---------------------------------

        st.subheader("Outlier Analysis")

        st.write("""
        The Interquartile Range (IQR) method identifies unusually
        low or high numerical values. Identifying an outlier does
        not automatically mean that the value is incorrect.
        """)

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        numeric_columns = [
            column for column in numeric_columns
            if column != "loan_id"
        ]

        outlier_results = []

        for column in numeric_columns:

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            count = (
                (df[column] < lower_limit) |
                (df[column] > upper_limit)
            ).sum()

            outlier_results.append({
                "Feature": column,
                "Outlier Count": int(count),
                "Lower Limit": round(lower_limit, 2),
                "Upper Limit": round(upper_limit, 2)
            })

        outlier_table = pd.DataFrame(outlier_results)

        st.dataframe(
            outlier_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Outliers are identified for analysis. "
            "This table does not claim that they were removed."
        )

        # ---------------------------------
        # 3. DATA PREPROCESSING
        # ---------------------------------

        st.header("3. Data Preprocessing")

        st.write("""
        Data preprocessing prepares the cleaned dataset for
        the machine learning algorithms.
        """)

        st.subheader("Feature Selection")

        st.write("""
        The loan_id column is excluded because it identifies an
        application rather than describing the applicant's financial
        situation. The loan_status column is separated as the target
        variable. The remaining applicant information is used as
        input features.
        """)

        if feature_columns:

            st.write("The model uses the following encoded features:")

            st.dataframe(
                pd.DataFrame({
                    "Model Input Features": feature_columns
                }),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Categorical Encoding")

        st.write("""
        Education and self-employment are categorical features.
        One-hot encoding converts these categories into numerical
        columns that the algorithms can use. The drop_first=True
        option removes the first category from each encoded feature.
        """)

        if categorical_columns:

            st.write(
                "Categorical columns: "
                + ", ".join(categorical_columns)
            )

        st.subheader("Numerical Features")

        if numerical_columns:

            st.write(
                "Numerical columns: "
                + ", ".join(numerical_columns)
            )

        st.subheader("Training and Testing Split")

        st.write("""
        The dataset is divided into 80% training data and 20%
        testing data. Stratified splitting helps preserve the
        distribution of approved and rejected applications in
        both sets.
        """)

        train_count = get_number(
            tuning_info, "training_records", 3415
        )

        test_count = get_number(
            tuning_info, "testing_records", 854
        )

        feature_count = get_number(
            tuning_info, "number_of_features", len(feature_columns)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Training Records", f"{train_count:,}")

        with col2:
            st.metric("Testing Records", f"{test_count:,}")

        with col3:
            st.metric("Input Features", feature_count)

        st.info(
            "The project uses random_state=42 to make the data split "
            "reproducible."
        )

        # ---------------------------------
        # 4. DATA VISUALISATIONS
        # ---------------------------------

        st.header("4. Data Visualisations")

        if "loan_status" in df.columns:

            st.subheader("Loan Status Distribution")

            status_counts = clean_status(
                df["loan_status"]
            ).value_counts()

            col1, col2 = st.columns(2)

            with col1:

                fig, ax = plt.subplots(figsize=(6, 4))

                status_counts.plot(
                    kind="bar",
                    ax=ax,
                    color=["#24527e", "#d5a52b"]
                )

                ax.set_title("Loan Status Distribution")
                ax.set_xlabel("Loan Status")
                ax.set_ylabel("Number of Applications")
                ax.tick_params(axis="x", rotation=0)

                fig.tight_layout()
                display_chart(fig)

            with col2:

                fig, ax = plt.subplots(figsize=(6, 4))

                status_counts.plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    ax=ax
                )

                ax.set_title("Loan Approval Percentage")
                ax.set_ylabel("")

                fig.tight_layout()
                display_chart(fig)

        # FINANCIAL ANALYSIS

        st.subheader("Financial Information Analysis")

        if "income_annum" in df.columns and "cibil_score" in df.columns:

            col1, col2 = st.columns(2)

            with col1:

                fig, ax = plt.subplots(figsize=(6, 4))

                df["income_annum"].plot(
                    kind="hist",
                    bins=20,
                    ax=ax,
                    color="#24527e"
                )

                ax.set_title("Annual Income Distribution")
                ax.set_xlabel("Annual Income")

                fig.tight_layout()
                display_chart(fig)

            with col2:

                fig, ax = plt.subplots(figsize=(6, 4))

                df["cibil_score"].plot(
                    kind="hist",
                    bins=20,
                    ax=ax,
                    color="#d5a52b"
                )

                ax.set_title("CIBIL Score Distribution")
                ax.set_xlabel("CIBIL Score")

                fig.tight_layout()
                display_chart(fig)

        # INCOME VS LOAN

        if "income_annum" in df.columns and "loan_amount" in df.columns:

            st.subheader("Income and Loan Amount Relationship")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.scatter(
                df["income_annum"],
                df["loan_amount"],
                alpha=0.5,
                color="#24527e"
            )

            ax.set_title("Annual Income vs Loan Amount")
            ax.set_xlabel("Annual Income")
            ax.set_ylabel("Loan Amount")

            fig.tight_layout()
            display_chart(fig)

        # DESCRIPTIVE STATISTICS

        st.subheader("Descriptive Statistics")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

        with st.expander("View Dataset Records"):

            st.dataframe(
                df,
                use_container_width=True
            )


# =========================================
# MODEL PERFORMANCE PAGE
# =========================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "This section explains model selection, training, tuning "
        "and evaluation."
    )

    # ---------------------------------
    # 1. MODEL DETERMINATION
    # ---------------------------------

    st.header("1. Model Determination")

    st.write("""
    Three machine learning classification algorithms were selected
    and compared for the loan prediction project.
    """)

    with st.expander("Logistic Regression - Why was it selected?"):

        st.write("""
        Logistic Regression was selected as a baseline classification
        algorithm. It is relatively simple and provides a useful
        starting point for comparing loan approval predictions.
        """)

    with st.expander("Decision Tree - Why was it selected?"):

        st.write("""
        Decision Tree was selected because it learns decision rules
        from applicant information. Its structure is relatively easy
        to understand and explain.
        """)

    with st.expander("Random Forest - Why was it selected?"):

        st.write("""
        Random Forest was selected because it combines predictions
        from multiple decision trees. This can improve predictive
        performance and reduce dependence on a single tree.
        """)

    # ---------------------------------
    # 2. MODEL TRAINING
    # ---------------------------------

    st.header("2. Model Training")

    st.write("""
    The three selected algorithms were trained using the same
    training dataset. The remaining records were reserved for testing.
    """)

    train_count = get_number(
        tuning_info, "training_records", 3415
    )

    test_count = get_number(
        tuning_info, "testing_records", 854
    )

    feature_count = get_number(
        tuning_info, "number_of_features", len(feature_columns)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Records", f"{train_count:,}")

    with col2:
        st.metric("Testing Records", f"{test_count:,}")

    with col3:
        st.metric("Input Features", feature_count)

    st.write("The models trained were:")

    st.markdown("""
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    """)

    # ---------------------------------
    # 3. MODEL COMPARISON
    # ---------------------------------

    st.header("3. Model Comparison")

    if comparison is not None:

        st.dataframe(
            comparison.round(2),
            use_container_width=True,
            hide_index=True
        )

        if (
            "Accuracy" in comparison.columns
            and "Model" in comparison.columns
        ):

            chart_data = comparison.copy()

            chart_data["Accuracy"] = pd.to_numeric(
                chart_data["Accuracy"],
                errors="coerce"
            )

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.bar(
                chart_data["Model"],
                chart_data["Accuracy"],
                color="#24527e"
            )

            ax.set_title("Model Accuracy Comparison")
            ax.set_xlabel("Machine Learning Model")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=15)

            fig.tight_layout()
            display_chart(fig)

        else:
            st.warning(
                "The model comparison file does not contain "
                "the expected Model and Accuracy columns."
            )

    else:

        st.warning(
            "Model comparison results are not available yet. "
            "Run model_experiments.py first."
        )

    # ---------------------------------
    # 4. MODEL TUNING
    # ---------------------------------

    st.header("4. Model Tuning")

    st.write("""
    Random Forest was further tuned using GridSearchCV.
    The tuning process tested different combinations of
    hyperparameters and used five-fold cross-validation.
    """)

    st.write("The parameter combinations included:")

    tuning_table = pd.DataFrame({
        "Parameter": [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf"
        ],
        "Values Tested": [
            "100, 200",
            "None, 10",
            "2, 5",
            "1, 2"
        ]
    })

    st.dataframe(
        tuning_table,
        use_container_width=True,
        hide_index=True
    )

    if tuning_info is not None:

        st.subheader("Tuning Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Cross-validation F1-score",
                f"{tuning_info.get('best_cross_validation_f1', 0):.2f}%"
            )

        with col2:

            st.metric(
                "Tuned Model Test Accuracy",
                f"{tuning_info.get('tuned_test_accuracy', 0):.2f}%"
            )

        st.write("Best Hyperparameters")

        parameter_labels = {
            "max_depth": "Maximum Depth",
            "min_samples_leaf": "Minimum Samples per Leaf",
            "min_samples_split": "Minimum Samples to Split",
            "n_estimators": "Number of Trees"
        }

        best_parameters = tuning_info.get(
            "best_parameters", {}
        )

        hyperparameter_table = pd.DataFrame([
            {
                "Hyperparameter": parameter_labels.get(
                    name,
                    name.replace("_", " ").title()
                ),
                "Selected Value": (
                    "Unlimited" if value is None else value
                )
            }
            for name, value in best_parameters.items()
        ])

        st.dataframe(
            hyperparameter_table,
            use_container_width=True,
            hide_index=True
        )

        st.info("""
        The best configuration used 100 decision trees, no fixed
        maximum depth, a minimum of two samples for splitting,
        and at least one sample per leaf.
        """)

    else:

        st.warning(
            "Tuning results are not available yet. "
            "Run model_experiments.py first."
        )

    # ---------------------------------
    # 5. MODEL EVALUATION
    # ---------------------------------

    st.header("5. Model Evaluation")

    if comparison is not None and "Model" in comparison.columns:

        tuned_row = comparison[
            comparison["Model"] == "Tuned Random Forest"
        ]

        if not tuned_row.empty:

            metrics = tuned_row.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.2f}%"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.2f}%"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.2f}%"
                )

            with col4:
                st.metric(
                    "F1-score",
                    f"{metrics['F1_Score']:.2f}%"
                )

        else:

            st.info(
                "Tuned Random Forest metrics will appear "
                "after the experiment has completed."
            )

    else:

        st.info(
            "Evaluation metrics will appear after model training."
        )

    # CLASSIFICATION REPORT

    st.subheader("Classification Report")

    if classification_results is not None:

        st.write("""
        The classification report provides precision, recall and
        F1-score for each loan status, as well as overall averages.
        """)

        st.dataframe(
            classification_results.round(3),
            use_container_width=True
        )

    else:

        st.warning("Classification report not found.")

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    if os.path.exists("confusion_matrix.png"):

        st.image(
            "confusion_matrix.png",
            caption="Tuned Random Forest Confusion Matrix",
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix has not been generated yet."
        )

    # FEATURE IMPORTANCE

    st.subheader("Feature Importance")

    if os.path.exists("feature_importance.png"):

        st.image(
            "feature_importance.png",
            caption="Top 10 Features Influencing Loan Predictions",
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance chart has not been generated yet."
        )

    # TUNING INTERPRETATION

    st.subheader("Interpretation of Model Results")

    st.write("""
    The Random Forest model achieved the highest test accuracy.
    The tuned Random Forest achieved the same test accuracy as
    the original Random Forest. This means that the selected
    hyperparameters did not improve the measured test performance.
    """)

    st.caption("""
    Model evaluation results are based on the project's held-out
    test dataset. They do not guarantee performance on future
    loan applications.
    """)


# =========================================
# FOOTER
# =========================================

st.markdown("""
<div class="footer">
    MOTZ FINANCIAL SERVICES &nbsp; | &nbsp;
    LOAN PREDICTION SYSTEM
</div>
""", unsafe_allow_html=True)
