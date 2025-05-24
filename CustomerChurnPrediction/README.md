# **🙍‍♂️ Customer Churn Prediction**

## 1. **Customer Churn Dataset**
**Notebook:** [Click here](https://github.com/julsCadenas/DS-ML-Projects/tree/main/CustomerChurnPrediction/notebooks/CustomerChurn/churn_model.ipynb)

### **Churn Model Performance Summary**
**Accuracy:** 82%  
**Churn detection rate (recall):** 91%  
**Precision for churn predictions:** 80%  
**F1-score for churn:** 0.85  
**AUC-ROC:** 0.9006  

### **Conclusion**
The model is now highly effective at identifying customers at risk of churning, capturing 91% of actual churners. This high recall is particularly valuable in churn prediction, as it minimizes missed opportunities for retention interventions. With 80% precision, the model maintains good reliability in its churn predictions, meaning retention resources won't be significantly wasted on false positives.

The optimal threshold of 0.022 is notably low, indicating that even customers with relatively small churn indicators should be flagged for attention. This aligns with the SHAP analysis showing that payment delays and support calls are strong churn predictors.

This balanced performance makes the model suitable for deployment in a real-world customer retention program, where it can effectively direct intervention efforts toward the right customers before they churn.


---

## 2. **Telco Customer Churn**
**Notebook:** [Click here](https://github.com/julsCadenas/DS-ML-Projects/tree/main/CustomerChurnPrediction/notebooks/TelcoChurn)

### **Churn Model Performance Summary**
**Best Model:** Random Forest (after hyperparameter tuning)
**Accuracy:** 93%
**Churn detection rate (recall):** 85%
**Precision for churn predictions:** 89%
**F1-score for churn:** 0.87

### **Conclusion**
After hyperparameter tuning and cross-validation, the Random Forest model showed excellent performance with a 93% accuracy and balanced precision-recall for churn detection. It successfully identifies 85% of churners while maintaining a precision of 89%, ensuring few false positives. The F1-score of 0.87 reflects strong overall performance in classifying churners.

The tuning process, which involved RandomizedSearchCV and cross-validation, led to slight but meaningful improvements over the base model. XGBoost also performed well, matching the overall accuracy and offering slightly better performance on the majority class, but Random Forest showed better precision-recall trade-offs on the minority churn class.

Feature engineering (e.g., adding Average Monthly Charge) and careful threshold analysis contributed to the model’s strong predictive power. These insights, alongside feature importance (to be visualized with SHAP), will guide targeted retention strategies and business decisions.

This model is now robust and reliable enough for deployment in customer service workflows, where early churn detection is critical for proactive retention efforts.

---