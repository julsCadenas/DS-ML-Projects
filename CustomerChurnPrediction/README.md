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
