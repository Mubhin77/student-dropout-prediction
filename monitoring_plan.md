# Model Monitoring Plan
## Student Dropout Prediction Using Machine Learning

### 1. Introduction

The purpose of this monitoring plan is to maintain the reliability and usefulness of the Student Dropout Prediction model after deployment. The model is deployed through a Streamlit application running inside a Docker container.

The current project uses historical student admission information to predict whether a student's eventual academic outcome is Graduate or Dropout. Monitoring is important because student characteristics and educational conditions may change over time.

### 2. Prediction Monitoring

The application should monitor the number of predictions made and the distribution of predicted outcomes.

The following information should be recorded:
- Total number of predictions.
- Number of Graduate predictions.
- Number of Dropout predictions.
- Average predicted dropout probability.
- Date and time of predictions.

A significant change in the proportion of Dropout predictions should trigger further investigation.

### 3. Data Drift Monitoring

Data drift occurs when the characteristics of new student data differ from the data used during model training.

The following features should be monitored:
- Age at enrollment.
- Admission grade.
- Previous qualification grade.
- Application mode.
- Course.
- Scholarship holder.
- Tuition fees up to date.

New student data should be compared with the original training data. Statistical methods such as the Population Stability Index (PSI) can help identify changes in feature distributions.

### 4. Model Performance Monitoring

Actual academic outcomes are not immediately available when admission information is collected. Therefore, model performance should be evaluated once verified student outcomes become available.

The following evaluation metrics should be monitored:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Precision-Recall AUC.

Special attention should be given to Dropout recall because failing to identify students who eventually drop out may limit the usefulness of the model.

### 5. Fairness Monitoring

The model should be reviewed to identify whether prediction errors differ across student groups.

Where legally and ethically appropriate, performance can be compared across groups based on gender, age, scholarship status, and international status.

Sensitive attributes should be handled responsibly, and predictions should never be used as the sole basis for educational decisions.

### 6. Model Retraining Strategy

The model should be reviewed periodically, such as every six months, or earlier if significant data drift or performance deterioration is identified.

Before retraining:
- Collect new verified student records.
- Check data quality and missing values.
- Review changes in feature distributions.
- Retrain and evaluate candidate models.
- Compare new results with the existing model.
- Replace the deployed model only after satisfactory evaluation.

All experiments and model versions should be documented using MLflow.

### 7. Monitoring Alerts

The following conditions should trigger an investigation:

| Monitoring area | Suggested alert condition |
|---|---|
| Prediction distribution | Unexpected increase in Dropout predictions |
| Data drift | PSI greater than 0.20 for important features |
| Model performance | Dropout recall falls below 0.60 |
| Application availability | Streamlit application becomes unavailable |
| Data quality | Missing or invalid input values |

These thresholds are initial project recommendations and should be reviewed before real-world use.

### 8. Current Project Limitations

The current project demonstrates model deployment but does not implement automated production monitoring, real-time drift detection, or automatic retraining.

The model was trained using historical admission information and evaluated on a held-out test dataset. It has not been validated as a live early-warning system. Monitoring activities described in this document are proposed future improvements.

### 9. Conclusion

A structured monitoring process can help maintain model reliability, identify changing student data patterns, and detect performance problems. Regular evaluation, responsible data handling, and controlled retraining are necessary before considering real-world educational use.