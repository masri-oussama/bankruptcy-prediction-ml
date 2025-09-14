# Bankruptcy Prediction Through ML Techniques

## Final Year Project - Lebanese University

A comprehensive machine learning approach to corporate bankruptcy prediction using financial ratios and advanced data science techniques.

## 📋 Project Overview

Corporate bankruptcy poses significant economic challenges, affecting investors, creditors, and the broader financial system. This project addresses this critical need by leveraging advanced machine learning techniques to build an early warning system for financial distress prediction.

## 🎯 Objectives

- Compare the effectiveness of various machine learning algorithms for bankruptcy prediction
- Evaluate different preprocessing techniques and their impact on model performance
- Analyze feature importance and selection methods for financial data
- Achieve superior predictive performance compared to traditional methods
- Provide actionable insights for financial risk assessment

## 📊 Dataset

- **Source:** Taiwan Economic Journal (1999-2009)
- **Total Samples:** 6,819 companies
- **Features:** 95 financial ratios
- **Target Distribution:** 
  - Bankrupt companies: 220 (3.2%)
  - Non-bankrupt companies: 6,599 (96.8%)

### Feature Categories
- **Profitability Ratios:** ROA, ROE, profit margins
- **Liquidity Ratios:** Current ratio, quick ratio
- **Leverage Ratios:** Debt-to-equity, debt-to-assets
- **Activity Ratios:** Asset turnover, inventory turnover

## 🔧 Methodology

### Data Preprocessing
- **Missing Value Treatment:** Imputation strategies
- **Outlier Detection:** Statistical methods and domain knowledge
- **Feature Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Class Balancing:** SMOTE (Synthetic Minority Oversampling Technique)

### Feature Selection
- **SelectKBest:** Statistical significance testing
- **Principal Component Analysis (PCA):** Dimensionality reduction
- **Recursive Feature Elimination:** Model-based selection

### Machine Learning Models
- **Traditional Methods:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes

- **Ensemble Methods:**
  - Random Forest
  - XGBoost
  - AdaBoost
  - Gradient Boosting

- **Deep Learning:**
  - Multi-layer Perceptron (MLP)
  - Deep Neural Networks with dropout and batch normalization

## 📈 Key Results

### Best Performing Models
1. **XGBoost:** 97.2% accuracy with SMOTE and hyperparameter tuning
2. **Random Forest:** 96.8% accuracy with robust ensemble performance
3. **SVM (RBF kernel):** 95.4% accuracy as the best traditional method

### Performance Metrics
- **Precision:** 94.8%
- **Recall:** 92.3%
- **F1-Score:** 93.5%
- **AUC-ROC:** 0.984

### Key Improvements
- **SMOTE Impact:** +18% improvement in minority class detection
- **Hyperparameter Tuning:** 8-15% performance gains
- **Feature Selection:** +12% improvement in precision

## 💡 Key Findings

1. **Machine Learning Superiority:** ML models significantly outperform traditional bankruptcy prediction methods
2. **Ensemble Excellence:** Random Forest and XGBoost achieve the highest accuracy (96-97%)
3. **Preprocessing Importance:** SMOTE balancing and feature selection are crucial for optimal results
4. **Feature Insights:** Net Income/Total Assets is the most predictive feature for bankruptcy

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Data Visualization:** matplotlib, seaborn
- **Class Balancing:** imbalanced-learn (SMOTE)
- **Development Environment:** Google Colab

## 📁 Repository Structure

```
bankruptcy-prediction-ml/
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── data/
│   ├── raw/
│   │   └── bankruptcy_data.csv
│   └── processed/
│       └── cleaned_data.csv
├── results/
│   ├── model_performance_comparison.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── docs/
│   └── thesis.pdf
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook or Google Colab
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/bankruptcy-prediction-ml.git
cd bankruptcy-prediction-ml
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Open the notebooks in Jupyter or upload to Google Colab:
```bash
jupyter notebook
```

## 📊 Usage

1. **Data Preprocessing:** Run `01_data_preprocessing.ipynb` to clean and prepare the data
2. **Exploratory Analysis:** Execute `02_exploratory_data_analysis.ipynb` for data insights
3. **Feature Selection:** Use `03_feature_selection.ipynb` to optimize features
4. **Model Training:** Run `04_model_training.ipynb` to train various ML models
5. **Evaluation:** Execute `05_model_evaluation.ipynb` to compare model performance

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

## 📞 Contact

For questions or collaboration opportunities, please reach out:
- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]

---

**Note:** This project was developed as part of the Final Year Project requirements for the BSc. in Data Science program at Lebanese University.

