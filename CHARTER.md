# Project Charter

## Early Warning System for Student Dropout Risk Using Machine Learning

**Project Domain:** Education  
**Project Type:** Supervised Binary Classification  
**Dataset:** Open University Learning Analytics Dataset (OULAD)  
**Development Language:** Python  
**Project Deadline:** 27 November 2026

---

## 1. Business Problem

Universities offering online and distance-learning courses may struggle to identify students who are at risk of withdrawing before they actually leave their courses.

Students may show signs of disengagement through reduced learning activity, irregular participation and limited interaction with online learning materials.

The purpose of this project is to develop an early warning system that uses historical student information and early learning engagement patterns to identify students who may withdraw from their courses.

## 2. Decision Changed

At the end of a student's first 28 days of course participation, an academic counsellor must decide whether the student should receive additional academic support.

The machine learning system will provide a risk prediction to help prioritise students for follow-up.

## 3. Decision Owner

The primary decision owner is an academic counsellor or student support officer responsible for student retention and academic intervention.

## 4. Proposed Action

Students identified as high risk may receive:

- Personalised academic guidance.
- Follow-up communication.
- Information about available learning resources.
- Additional academic support.

The model will support human decision-making and will not automatically determine a student's academic status.

## 5. Machine Learning Task

Binary classification.

- Class 1: Student withdraws after the prediction point.
- Class 0: Student does not withdraw after the prediction point.

## 6. Target Variable

The target will be derived from the `final_result` field in `studentInfo.csv`, validated against `date_unregistration` in `studentRegistration.csv`.

Students who withdrew on or before day 28 will be excluded.

Students who withdraw after day 28 will be labelled 1. Eligible students who do not withdraw will be labelled 0.

## 7. Unit of Prediction

One student-course registration.

The prediction will be generated at the end of day 28, using information available during days 0–27 inclusive.

## 8. Baselines

Three baseline approaches will be evaluated:

1. Majority-class classifier.
2. Simple low-engagement rule based on early VLE activity.
3. Logistic Regression using early student engagement and registration features.

Random Forest will be evaluated as an additional model family.

## 9. Evaluation Criteria

Primary metric: Precision-Recall Area Under the Curve (PR-AUC).

Operational metric: Recall among the highest-risk 20% of eligible students.

Proposed success criterion:

- The selected model should improve recall at 20% intervention capacity by at least five percentage points over the simple rule baseline.
- The model should outperform the majority-class baseline.
- Cross-validation performance should be reasonably consistent.
- No information recorded after the prediction point may be used as a feature.

The 20% intervention capacity is a hypothetical operational assumption for this academic project.

## 10. Business Success

The intended benefit is to help academic support teams prioritise students who may need additional assistance.

The operational benefit will be assessed using the proportion of future withdrawals identified within the assumed intervention capacity.

No actual financial savings or causal improvement in student retention will be claimed without a real-world intervention study.

## 11. Guardrails

- No future information or target leakage.
- No automatic punitive decisions.
- Demographic attributes will not be used as model predictors in the initial model.
- Performance differences across demographic groups will be examined.
- Predictions will be treated as risk estimates rather than certainties.
- Human review is required before any intervention.

## 12. Kill Criteria

Deployment will not be recommended if:

- The model does not meaningfully outperform the simple rule baseline.
- Performance is unstable across validation folds.
- Data leakage cannot be ruled out.
- Significant and unexplained group disparities remain.
- The system cannot produce reliable and reproducible predictions.

## 13. Known Risks and Limitations

- OULAD represents UK distance-learning students, not Nepalese Montessori students.
- Historical engagement patterns may not represent current student behaviour.
- Some students may engage through activities not captured by the VLE.
- Historical correlations do not establish that an intervention will prevent withdrawal.
- Student demographics and course differences may affect model performance.

## 14. Recommendation Policy

The final recommendation will be one of the following:

- Deploy: if all agreed performance, fairness and reliability criteria are met and an appropriate human review process exists.
- Do not deploy: if the model fails the agreed criteria.
- Further validation required: if the model shows promise but needs additional evidence.

**Charter status:** Initial project charter, established before modelling.
