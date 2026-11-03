# Bank Marketing Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit following a marketing campaign.

The project implements five classification algorithms using a common dataset and compares their performance using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit application is also developed to allow users to upload labelled test data, select a trained classification model, and view its evaluation results.

## 2. Dataset Description

**Dataset:** Bank Marketing (Bank Marketing Campaign Dataset)

**Source:** UCI Machine Learning Repository

**Citation:**

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

**Dataset link:** https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

The selected dataset contains information about direct marketing campaigns conducted by a Portuguese banking institution.

| Dataset property | Description |
|---|---|
| Total records | 41,188 |
| Original columns | 21 |
| Input features used | 19 |
| Target column | y |
| Classification type | Binary classification |
| Target classes | Yes / No |

The target variable `y` indicates whether a customer subscribed to a term deposit.

The target distribution is:

| Class | Records | Percentage |
|---|---:|---:|
| No | 36,548 | 88.73% |
| Yes | 4,640 | 11.27% |

The dataset is imbalanced, with substantially more negative examples than positive examples.

## 3. Data Preprocessing

The original dataset contains 20 input features and one target column.

The `duration` feature was excluded from model training because it represents the duration of the marketing call. This information is generally unavailable before the call and could introduce information leakage when predicting subscription outcomes.

The remaining 19 input features were divided into numerical and categorical variables.

**Numerical features:**
- age
- campaign
- pdays
- previous
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

**Categorical features:**
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- poutcome

The following preprocessing operations were applied:

1. Numerical features were standardized using `StandardScaler`.
2. Categorical features were converted into numerical representations using `OneHotEncoder`.
3. `handle_unknown="ignore"` was used to handle previously unseen categorical values during prediction.
4. The target variable was encoded as 0 for No and 1 for Yes.
5. The dataset was divided into training and testing subsets using an 80:20 stratified split.

**Train-test split:**

| Subset | Records |
|---|---:|
| Training | 32,950 |
| Testing | 8,238 |

The split used `random_state=42` and stratification to preserve the target class distribution.

The original class imbalance was retained. No resampling technique was applied.

## 4. Machine Learning Models

Five classification algorithms were implemented using Scikit-learn.

### 4.1 Logistic Regression

Logistic Regression was used as a linear classification baseline. It estimates the probability of a customer subscribing to a term deposit.

### 4.2 Decision Tree

A Decision Tree classifier was implemented with a maximum depth of 10. It learns decision rules by recursively splitting the feature space.

### 4.3 K-Nearest Neighbors (KNN)

KNN was implemented with five neighbors. It classifies observations based on the labels of nearby training observations.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was used as a probabilistic classification algorithm. A dense preprocessing pipeline was used for this model.

### 4.5 Linear Support Vector Machine

A Linear SVM classifier was implemented using `LinearSVC` with C=1.0.

For AUC evaluation, the model's decision scores were used because `LinearSVC` does not directly provide probability estimates.

All preprocessing steps and classifiers were combined into pipelines and saved as trained model artifacts.

## 5. Model Evaluation and Comparison

All five classifiers were evaluated using the same held-out test dataset containing 8,238 records.

The following metrics were calculated:

- **Accuracy:** Overall proportion of correctly classified observations.
- **AUC:** Ability of the classifier to distinguish between the two target classes across different decision thresholds.
- **Precision:** Proportion of predicted positive observations that were actually positive.
- **Recall:** Proportion of actual positive observations correctly identified.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **MCC:** Correlation-based measure of classification quality that accounts for all four confusion matrix outcomes.

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.9013 | 0.7581 | 0.6353 | 0.2909 | 0.3991 | 0.3856 |
| KNN | 0.8972 | 0.7443 | 0.5839 | 0.3039 | 0.3997 | 0.3719 |
| Gaussian Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Linear SVM | 0.9014 | 0.7979 | 0.7164 | 0.2069 | 0.3211 | 0.3502 |

## 6. Observations and Model Analysis

The experimental results show that the five classifiers have different strengths.

**Logistic Regression**
- Achieved the highest AUC of 0.8008.
- Provided a relatively strong balance between accuracy and precision.
- Its recall was comparatively low at 0.2188.

**Decision Tree**
- Achieved the highest MCC of 0.3856.
- Obtained an F1 Score of 0.3991.
- Provided a comparatively balanced classification result among the models with higher accuracy.

**K-Nearest Neighbors**
- Achieved an F1 Score of 0.3997.
- Its recall was higher than Logistic Regression and Linear SVM.
- Its AUC was the lowest among the five models at 0.7443.

**Gaussian Naive Bayes**
- Achieved the highest recall of 0.6347.
- Also achieved the highest F1 Score of 0.4230.
- Its precision was relatively low at 0.3172, indicating a higher number of false-positive predictions.

**Linear SVM**
- Achieved the highest accuracy of 0.9014.
- Also achieved the highest precision of 0.7164.
- Its recall was the lowest among the five models at 0.2069.

### Overall Model Comparison

There is no single classifier that performs best across all evaluation metrics.

Gaussian Naive Bayes achieved the highest F1 Score and recall, making it useful when identifying more potential subscribers is important.

Decision Tree achieved the highest MCC, while Logistic Regression obtained the highest AUC.

Linear SVM achieved the highest accuracy and precision.

Since the dataset is imbalanced, accuracy alone is not sufficient to determine the best classifier. The choice of model depends on the objective and the relative importance of identifying potential subscribers versus avoiding false-positive predictions.

These results represent baseline performance using the selected model configurations and default classification thresholds. Further experimentation with decision thresholds and class-imbalance handling may improve performance.

## 7. Streamlit Application

A Streamlit application was developed to evaluate the saved classification models through an interactive interface.

The application provides the following functionality:

1. Upload a labelled CSV test dataset.
2. Select one of the five trained classification models.
3. Load the saved model artifact without retraining.
4. Generate predictions for the uploaded records.
5. Calculate Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
6. Display the confusion matrix.
7. Display the classification report.
8. Preview the uploaded dataset.

The application uses the saved preprocessing and classification pipelines, ensuring that uploaded data undergoes the same transformations used during model training.

## 8. Project Structure

```text
bank_marketing_classification_repo/
│
├── bank_marketing_classification.ipynb
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── models/
    └── trained_models.joblib
```

## 9. Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/Aravind-DEV83/bank_marketing_classification_repo.git
```

Move into the project directory:

```bash
cd bank_marketing_classification_repo
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit and upload the labelled test dataset to evaluate the selected model.

## 10. Repository and Deployment Links

**GitHub Repository:**

https://github.com/Aravind-DEV83/bank_marketing_classification_repo

**Live Streamlit Application:**

[Streamlit application](https://bank-customer-classification-05462.streamlit.app/)

---

**Dataset Reference**

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.