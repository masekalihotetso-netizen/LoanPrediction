 LOAN PREDICTION SYSTEM USING MACHINE LEARNING
 MOTZ FINANCIAL SERVICES

Limkokwing University of Creative Technology 
Machine Learning Group Assignment

---

 1. Project Overview

The Loan Prediction System is a machine learning project developed for Motz Financial Services. The system uses historical loan application data to predict whether a loan application is likely to be Approved or Rejected.

The project applies supervised machine learning algorithms to identify patterns in previous loan applications and use those patterns to make predictions.

A Streamlit dashboard was also developed to allow users to explore the dataset, view model performance and enter applicant information to receive a loan prediction.

2. Project Objectives

The main objectives of this project are:

- To collect and examine the loan approval dataset.
- To identify and handle missing values.
- To analyse the dataset using descriptive statistics and visualisations.
- To prepare the data for machine learning.
- To train and compare different classification algorithms.
- To tune the selected model to evaluate its performance.
- To develop an interactive dashboard for loan predictions.
- To evaluate the models using appropriate performance metrics.
3. Dataset Description

The project uses the **Loan Approval Prediction Dataset**.

The dataset contains 4,269 loan application records and 13 original columns. The `loan_id` column was removed during preparation because it is an identification field rather than a useful prediction feature.

The cleaned dataset contains 12 columns:

- 11 input features.
- 1 target column (`loan_status`).

 Input Features

| Feature | Description |
|---|---|
| no_of_dependents | Number of dependents |
| education | Applicant's education status |
| self_employed | Whether the applicant is self-employed |
| income_annum | Applicant's annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan repayment period |
| cibil_score | Applicant's credit score |
| residential_assets_value | Value of residential assets |
| commercial_assets_value | Value of commercial assets |
| luxury_assets_value | Value of luxury assets |
| bank_asset_value | Value of bank assets |

Target Variable

The `loan_status` column represents the loan decision:

- Approved
- Rejected

The dataset contains:

- 2,656 Approved applications (62.22%).
- 1,613 Rejected applications (37.78%).

4. Technologies Used

The following technologies and libraries were used:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Streamlit | Interactive dashboard development |
| Joblib | Saving and loading the trained model |
| Visual Studio Code | Development environment |

 5. Data Preprocessing

Data preprocessing was performed to prepare the dataset for machine learning.

The main preprocessing activities included:

1. Loading the dataset.
2. Inspecting the dataset structure and data types.
3. Removing the `loan_id` identification column.
4. Checking for missing values.
5. Filling missing numerical values using the median.
6. Filling missing categorical values using the most frequent category.
7. Converting categorical information into numerical form using one-hot encoding.
8. Separating the input features from the target variable.
9. Splitting the dataset into training and testing sets.

The dataset was divided using an 80:20 stratified split:

- Training records: 3,415.
- Testing records: 854.

The split used `random_state=42` to support reproducibility.

Preprocessing limitation: In the current experiment script, missing-value handling and categorical encoding are performed before the train-test split. This creates a possibility of data leakage. A future improvement would be to fit preprocessing steps only on the training data using a Scikit-learn Pipeline.

6. Machine Learning Models

Three machine learning algorithms were selected and evaluated.

6.1 Logistic Regression

Logistic Regression was used as a baseline classification algorithm. It estimates the probability that a loan application belongs to a particular class.

 6.2 Decision Tree

Decision Tree classifies applications by following a series of decision rules based on applicant information.

6.3 Random Forest

Random Forest combines multiple decision trees to produce a final prediction. It was selected because it can capture more complex relationships in the dataset.

The three algorithms were compared to identify the model with the strongest performance.

7. Hyperparameter Tuning

Hyperparameter tuning was performed on the Random Forest model using `GridSearchCV`.

The purpose was to test different parameter combinations and identify a suitable configuration.

The tuning process used:

- 5-fold cross-validation.
- 16 parameter combinations.
- 80 total model fits.

 Best Parameters

| Parameter | Selected Value |
|---|---|
| n_estimators | 100 |
| max_depth | None |
| min_samples_split | 2 |
| min_samples_leaf | 1 |

The best cross-validation F1-score was approximately 98.15%.

The tuned Random Forest was selected as the final saved model.

Tuning did not increase the held-out test accuracy compared with the original Random Forest. However, it provided a systematic way to evaluate different parameter combinations.

 8. Model Performance

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score

The following results were obtained on the 854-record test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.80% | 81.01% | 80.80% | 80.13% |
| Decision Tree | 97.07% | 97.07% | 97.07% | 97.07% |
| Random Forest | 98.13% | 98.15% | 98.13% | 98.12% |
| Tuned Random Forest | 98.13% | 98.15% | 98.13% | 98.12% |

 Selected Model

The Tuned Random Forest was selected as the final model because it achieved the highest test accuracy of 98.13% among the evaluated models.

The results show that Random Forest performed better than Logistic Regression and Decision Tree on this test set.

These results represent performance on the project's internal test dataset. They do not guarantee the same accuracy on new loan applications.

 9. Dashboard Features

An interactive dashboard was developed using Streamlit.

The dashboard contains four main pages:

9.1 Home Page

Provides an introduction to the loan prediction system and an overview of the project.
 9.2 Prediction Page

Allows users to enter applicant information and receive a predicted loan decision.

 9.3 Data Analysis Page

Displays information about the dataset, including loan status distribution and other visualisations.

 9.4 Model Performance Page

Displays model comparison results, evaluation metrics and model performance visualisations.

10. Project Files

The main project files are organised as follows:

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard application |
| `model_experiments.py` | Model training, comparison and hyperparameter tuning |
| `train_model.py` | Earlier model training script |
| `clean_data.py` | Data cleaning script |
| `preprocess_data.py` | Data preprocessing script |
| `inspect_dataset.py` | Dataset inspection script |
| `model_visualizations.py` | Model visualisation script |
| `loan_approval_dataset.csv` | Original dataset |
| `cleaned_loan_dataset.csv` | Cleaned dataset |
| `best_loan_model.joblib` | Saved trained model and supporting feature information |
| `model_comparison.csv` | Model comparison results |
| `model_tuning_results.json` | Hyperparameter tuning results |
| `classification_report.csv` | Classification evaluation results |
| `model_accuracy_comparison.png` | Model accuracy comparison chart |
| `confusion_matrix.png` | Confusion matrix visualisation |
| `feature_importance.png` | Feature importance visualisation |
| `loan_status_distribution.png` | Loan status distribution chart |

The saved `best_loan_model.joblib` file contains the trained model and supporting information, including feature columns, categorical and numerical column names, category levels and the target column name. This information helps the dashboard prepare applicant inputs in the correct format.

11. How to Run the Project

 11.1 Requirements

The project requires Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

 11.2 Install Dependencies

Open the project folder in Visual Studio Code and use the PowerShell terminal.

If the virtual environment has already been created, install the libraries using:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

 11.3 Train and Evaluate the Models

Run the model experiment script:

```powershell
.\.venv\Scripts\python.exe model_experiments.py
```

This script trains and evaluates the machine learning models, performs hyperparameter tuning and saves the model outputs.

 11.4 Run the Dashboard

Start the Streamlit application using:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard will open in a web browser.

Important: The dashboard requires the saved model file `best_loan_model.joblib` and its supporting project files to be available in the project folder.

 12. Limitations and Future Improvements

Although the system achieved good results, some limitations should be considered:

- The model was trained using one dataset.
- The dataset does not contain every possible factor that may affect loan approval.
- The current preprocessing process has a possible data leakage limitation.
- The model's performance on new applications has not yet been independently verified.
- Machine learning predictions should not replace proper financial assessment.

Future improvements could include:

1. Using a larger and more diverse dataset.
2. Fitting preprocessing steps only on training data.
3. Testing the model on independent external data.
4. Improving the dashboard with additional reporting features.
5. Monitoring model performance over time.
6. Assessing possible bias and fairness across applicant groups.

13. Conclusion

The project successfully developed a machine learning-based loan prediction system for Motz Financial Services. Three classification algorithms were evaluated, and Random Forest achieved the strongest test performance. The tuned Random Forest was saved and integrated into a Streamlit dashboard that allows users to explore loan data, view model performance and make predictions. The system demonstrates how machine learning can support loan assessment, although predictions should be reviewed by qualified personnel before financial decisions are made.

 14. Academic Information

Institution: Limkokwing University of Creative Technology

Assignment: Loan Prediction using Machine Learning

Client: Motz Financial Services

Course: Machine Learning

Project Type: Group Assignment

Project Repository:  
[Loan Prediction System on GitHub](https://github.com/mputanarele/LoanPrediction)

Group Responsibilities

| Group Member | Assigned Responsibilities |
|---|---|
| Mpho Nts'olo | Data gathering, data cleaning, missing values, outlier analysis, descriptive statistics and visualisations |
| Lihotetso Maseka | Data preprocessing, feature preparation, model selection, training and tuning |
| Relebohile Mputana | Streamlit dashboard development, model integration, testing, documentation and presentation |




