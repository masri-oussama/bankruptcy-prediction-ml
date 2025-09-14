# Bankruptcy Prediction Using Machine Learning

This project aims to build a predictive system for identifying companies at risk of bankruptcy using financial ratios and modern machine learning techniques.

## Dataset

The **Taiwanese Bankruptcy Prediction dataset** was used in this project, which contains **6819 companies' financial data over a period of 10 years (1999–2009)**. The dataset includes **64 financial ratios**, with the target variable indicating whether the company went bankrupt.

- **Class Imbalance:** Only **3%** of companies in the dataset are labeled as bankrupt, which required handling techniques such as **SMOTE (Synthetic Minority Oversampling Technique)**.
- **Target Variable:** `Bankrupt?` (1 = Bankrupt, 0 = Not bankrupt)

## Objective

The main goals of this project are:

1. Identify the financial ratios that are most effective in predicting corporate bankruptcy.
2. Evaluate the performance of various machine learning models under **realistic class-imbalance conditions**.

## Methods and Models

Applied a range of machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)
  
To improve performance and interpretability, the following techniques were applied:

- **Feature selection** via `SelectKBest`
- **Oversampling** using `SMOTE`
- **Hyperparameter tuning** 

## Evaluation Metrics

Due to the severe class imbalance, **accuracy alone is misleading** (even though some models reached up to **96% accuracy**). Therefore, the main focus was on:

- **Precision** for the bankrupt class (minimizing false positives)
- **Recall** for the bankrupt class (minimizing false negatives)
- **F1-score** to balance both
- ROC-AUC and Precision-Recall curves for further insights

## Key Findings

The top-performing models in predicting bankrupt companies were:

**Logistic Regression**: F1-score ≈ **49%**
**XGBoost**: F1-score ≈ **47.5%**
**Random Forest**: F1-score ≈ **45%**

These results were obtained after applying **SelectKBest** for feature selection, **SMOTE** for handling class imbalance, and **hyperparameter tuning** to optimize model performance.

## ROC Curve Comparison

The ROC curves below show each model's ability to distinguish between bankrupt and non-bankrupt companies. Most models performed well, with AUC scores over 0.9.

![ROC Curves](https://github.com/user-attachments/assets/515c1a4f-6411-45b8-b1f0-5ab851a50a4d)


---

## Model Performance Summary

While overall accuracy was high, it does not reflect the model’s ability to detect bankruptcies. The F1-score for class 1 (bankrupt) shows more realistic performance.

![image](https://github.com/user-attachments/assets/8f27b599-05ee-41eb-bc46-7491e630f8a6)


## References

- [Taiwanese Bankruptcy Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction?resource=download) – UCI Machine Learning Repository

## 🔮 Future Work

- **Cross-Market Validation:** Test models on different geographic markets
- **Real-Time Implementation:** Develop streaming prediction systems
- **Multi-Modal Data:** Incorporate text, news sentiment, and market indicators
- **Explainable AI:** Implement interpretability methods for model decisions
- **Deep Learning Enhancement:** Explore advanced neural network architectures

## 📝 Applications

- **Investment Risk Assessment:** Help investors evaluate company financial health
- **Credit Decision Making:** Assist lenders in loan approval processes
- **Regulatory Compliance:** Support financial regulators in monitoring systemic risks
- **Corporate Management:** Enable early intervention for financial distress

## 👥 Authors

- **Oussama EL MASRI** - Data Science Student, Lebanese University
- **Ali HUSSEIN** - Data Science Student, Lebanese University

## 👨‍🏫 Supervisors

- **Dr. Elie DINA** - Lebanese University
- **Dr. Kassem RAMMAL** - Lebanese University

## 🏫 Institution

**Lebanese University**  
Faculty of Information II  
BSc. in Data Science  
June 2025

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Lebanese University for providing research facilities and support
- Taiwan Economic Journal for the bankruptcy dataset
- Open-source community for the machine learning libraries
- Our supervisors for their invaluable guidance and expertise

---
**Note:** This project was developed as part of the Final Year Project requirements for the BSc. in Data Science program at Lebanese University.