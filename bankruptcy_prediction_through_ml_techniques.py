
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


## EDA

df = pd.read_csv('/Users/oussamaelmasri/Documents/MLP/Original data.csv')

# Check the first 5 rows of the data
df.head()

# Check the last 5 rows of the data
df.tail()

# Overview of the dataset dimensions
df.shape

# Check column names and data types
df.info()

# Basic Statistics
df.describe()

# Missing values check
df.isnull().sum()

# Heatmap for the missing values across all the columns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap (None should be visible)")
plt.show()

# Duplicate check
df.duplicated().sum()

# Inspection of the class distribution
df['Bankrupt?'].value_counts(normalize=True)

"""The result above shows a **severe** class imbalance in the target variable."""

# Analyze column value distributions and convert low-cardinality numeric columns to object

# First, display normalized value counts (percentages) for each column
for col in df.columns:
    print(f"\nNormalized value counts for column: '{col}'")
    display(df[col].value_counts(normalize=True, dropna=False).to_frame('percentage'))

# Second, convert numeric columns with ≤ 10 unique values to object (categorical-like)
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].nunique(dropna=False) <= 10:
        df[col] = df[col].astype('object')

# Show updated column type distribution
print("\nUpdated column type counts:")
print(df.dtypes.value_counts())

def plot_categorical_columns(df):
    """
    Plots bar and pie charts for all categorical (object) columns in the DataFrame.
    """
    cat_columns = df.select_dtypes(include='object').columns.tolist()

    for col in cat_columns:
        counts = df[col].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar Plot
        counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title(f'Bar Plot of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Pie Chart
        axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        axes[1].set_title(f'Pie Chart of {col}')
        axes[1].axis('equal')

        plt.tight_layout()
        plt.show()

# Plot only the remaining categorical columns
plot_categorical_columns(df)

# Drop irrelevant categorical columns

df = df.drop(['Liability-Assets Flag', 'Net Income Flag'], axis=1, errors='ignore')

"""Inspection of outliers"""

# Get number of numeric columns
num_cols = len(df.select_dtypes(include='number').columns)
cols_per_row = 4
num_rows = math.ceil(num_cols / cols_per_row)

# Create boxplots with dynamic layout
df.select_dtypes(include='number').plot(
    kind='box',
    subplots=True,
    layout=(num_rows, cols_per_row),
    figsize=(cols_per_row * 4, num_rows * 4),
    sharey=False
)

plt.suptitle('Outlier Visualization with Boxplots', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title
plt.show()

"""Inspection of correlation in the data"""

# Compute correlation matrix on cleaned data
corr_matrix = df.corr().abs()

# Plot the heatmap
plt.figure(figsize=(60, 45))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={'label': 'Correlation Coefficient'}
)

# Adjust axis labels and font size
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

target_column = df.columns[0]

# Calculate correlation with target variable
correlations = df.corr()[target_column].abs().sort_values(ascending=False)

# Remove the target variable itself from the correlations
correlations = correlations.drop(target_column)

# Get top 10 features most correlated with target
top_10_features = correlations.head(10).index.tolist()

print("Top 10 Features Most Correlated with Target Variable:")
print("=" * 55)
for i, feature in enumerate(top_10_features, 1):
    corr_value = correlations[feature]
    print(f"{i:2d}. {feature}: {corr_value:.4f}")

# Plotting the correlated features in a heatmap
corr_matrix = df[top_10_features].corr().abs()
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.title("Correlation Heatmap of Top 10 Features", fontsize=14)
plt.tight_layout()
plt.show()

# Function to find highly correlated feature pairs
def find_highly_correlated(corr_matrix, threshold=0.9):
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    return correlated_pairs

# Apply the function to detect highly correlated pairs
high_corr_pairs = find_highly_correlated(corr_matrix, threshold=0.9)

# Display the results
print("\n Highly Correlated Feature Pairs (>|0.9|):\n")
for f1, f2, corr_value in high_corr_pairs:
    print(f"- {f1} and {f2}: Correlation = {corr_value:.2f}")

# Get the correlation values for the top 10 features (sorted in decreasing order)
top_10_correlations = correlations.head(10)

# Create horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(top_10_correlations)), top_10_correlations.values,
                color='skyblue', edgecolor='navy', linewidth=0.7)

# Customize the plot
plt.xlabel('Correlation Coefficient (Absolute Value)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title(f'Top 10 Features Most Correlated with {target_column}', fontsize=14, fontweight='bold')

# Set y-axis labels (features names)
plt.yticks(range(len(top_10_correlations)), top_10_correlations.index, fontsize=10)

# Add correlation values on the bars
for i, (feature, corr_val) in enumerate(top_10_correlations.items()):
    plt.text(corr_val + 0.01, i, f'{corr_val:.3f}',
             va='center', ha='left', fontsize=10, fontweight='bold')

# Add a grid for better readability
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Invert y-axis to show highest correlation at the top
plt.gca().invert_yaxis()

# Adjust layout and display
plt.tight_layout()
plt.show()

"""# Preprocessing"""

df_normalized = df.copy()

# Select only numerical columns
num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Initialize the scaler
scaler = StandardScaler()

# Apply Min-Max normalization
df_normalized[num_columns] = scaler.fit_transform(df[num_columns])

"""Handling outliers using std capping"""

threshold = 4
df_capped = df_normalized.copy()

for col in df_capped.select_dtypes(include='number').columns:
    mean = df_capped[col].mean()
    std = df_capped[col].std()
    upper = mean + threshold * std
    lower = mean - threshold * std

    df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)

"""Comparison of Data Before and After Normalization and Outlier Handling"""

# List of numeric columns
numeric_cols = df_normalized.select_dtypes(include='number').columns

# Loop through each column and plot 'before' and 'after' side-by-side
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before handling outliers
    df_normalized.boxplot(column=col, ax=axes[0])
    axes[0].set_title(f'{col} - Before Handling')

    # After handling outliers (standard deviation capping)
    df_capped.boxplot(column=col, ax=axes[1])
    axes[1].set_title(f'{col} - After Handling')

    plt.suptitle(f'Before/After Comparison for "{col}"', fontsize=16)
    plt.tight_layout()
    plt.show()

"""Handling Multicollinearity"""

def find_highly_correlated(corr_matrix, threshold=0.9):
    correlated_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i):
            corr_val = corr_matrix.iloc[i, j]
            if pd.notna(corr_val) and abs(corr_val) > threshold:
                correlated_pairs.append((cols[i], cols[j], corr_val))
    return correlated_pairs

def drop_correlated_features(df_clean, features, target_col='Bankrupt?', corr_threshold=0.9):
    # Compute correlation matrix for features only
    corr_matrix = df_clean[features].corr().abs()
    correlated_pairs = find_highly_correlated(corr_matrix, threshold=corr_threshold)

    to_drop = set()

    for f1, f2, corr_value in correlated_pairs:
        # Compute correlation with target, drop NaNs
        corr_with_target_f1 = abs(df_clean[[f1, target_col]].dropna().corr().iloc[0, 1])
        corr_with_target_f2 = abs(df_clean[[f2, target_col]].dropna().corr().iloc[0, 1])

        # Drop the feature with lower correlation with target
        if corr_with_target_f1 > corr_with_target_f2:
            to_drop.add(f2)
        else:
            to_drop.add(f1)

    cleaned_features = [f for f in features if f not in to_drop]

    print(f"\nDropped features due to high correlation: {list(to_drop)}")
    print(f"Remaining features after cleaning: {cleaned_features}")

    return cleaned_features

final_features = drop_correlated_features(df_capped, top_10_features, target_col='Bankrupt?', corr_threshold=0.9)

# Get the list of dropped features from your multicollinearity analysis
dropped_features = [' Net worth/Assets',
                   ' ROA(C) before interest and depreciation before interest',
                   ' Per Share Net profit before tax (Yuan ¥)',
                   ' ROA(A) before interest and % after tax',
                   ' Net profit before tax/Paid-in capital',
                   ' ROA(B) before interest and depreciation after tax']

# Create 'df_final' by dropping the highly correlated features
df_final = df_capped.drop(columns=dropped_features)

# Display information about the final dataframe
print("Original DataFrame shape:", df_capped.shape)
print("Final DataFrame shape:", df_final.shape)
print(f"Features dropped: {len(dropped_features)}")
print(f"Features remaining: {df_final.shape[1]}")

print("\nDropped features:")
for i, feature in enumerate(dropped_features, 1):
    print(f"{i}. {feature}")
# This is the final dataframe that is ready to use in modeling

"""# Modeling

"""

# 1. Prepare Data
X1 = df_final.select_dtypes(include=['int64', 'float64'])
y1 = df_final['Bankrupt?'].astype(int)

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)

# 3. Applying SelectKBest with ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X1, y1)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y1, test_size=0.2, random_state=42, stratify=y1
)

# 5. SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 6. Compute scale_pos_weight for XGB
neg, pos = np.bincount(y_res)
scale_pos_weight = neg / pos

# 7. Define model sets
models_tuned = {
    'LR': LogisticRegression(C=1, solver='liblinear', max_iter=1000),
    'XGB': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         scale_pos_weight=scale_pos_weight, eval_metric='logloss'),
    'SVM': SVC(C=1, kernel='rbf', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=None),
    'DT': DecisionTreeClassifier(),
    'ANN': MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                         alpha=1e-4, max_iter=500)
}

models_default = {
    'LR': LogisticRegression(),
    'XGB': XGBClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(),
    'DT': DecisionTreeClassifier(),
    'ANN': MLPClassifier(max_iter=300)
}

# 8. Evaluation Function
def evaluate_models(models_dict, label):
    scores = []
    for name, model in models_dict.items():
        model.fit(X_res, y_res)
        prob = model.predict_proba(X_test)[:,1]
        prec, rec, th = precision_recall_curve(y_test, prob)
        f1s = 2*prec*rec / (prec+rec+1e-8)
        idx = np.argmax(f1s[:-1])
        thresh = th[idx]
        preds = (prob >= thresh).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"\n--- {label} {name} (th={thresh:.3f}) ---")
        print(f"Accuracy: {acc:.2%},  F1: {f1s[idx]:.3f}")
        print(classification_report(y_test, preds))
        scores.append((name, label, acc, f1s[idx]))
    return scores

# 9. Run evaluations
scores_tuned = evaluate_models(models_tuned, 'Tuned')
scores_default = evaluate_models(models_default, 'Default')

# 10. Combine results
df_scores = pd.DataFrame(scores_tuned + scores_default,
                         columns=['Model', 'Type', 'Accuracy', 'F1'])

# 11. Plot Accuracy and F1 Score
for metric in ['Accuracy', 'F1']:
    df_plot = df_scores.pivot(index='Model', columns='Type', values=metric)
    ax = df_plot.plot(kind='bar', figsize=(10, 6), title=f'{metric} Comparison: Tuned vs Default')
    ax.set_ylabel(metric)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Model Type')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Plot ROC Curves
plt.figure(figsize=(12, 8))
for name, model in models_tuned.items():
    # Predict probabilities
    probas = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Diagonal line for random classifier
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Plot settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Tuned Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data from tuned models
models = ['LR', 'XGB', 'SVM', 'KNN', 'RF', 'DT', 'ANN']
f1_scores = [0.490, 0.475, 0.374, 0.364, 0.448, 0.331, 0.329]
accuracies = [0.9619, 0.9531, 0.9509, 0.9487, 0.9531, 0.9289, 0.9582]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Color schemes
f1_color = '#5DADE2'  # Light blue
accuracy_color = '#58D68D'  # Light green

# Plot F1-Score (Class 1 - Bankrupt)
bars1 = ax1.bar(models, f1_scores, color=f1_color, alpha=0.8, edgecolor='white', linewidth=1)
ax1.set_title('F1-Score (Class 1 - Bankrupt)', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('F1-Score', fontsize=12)
ax1.set_ylim(0, 0.5)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, value in zip(bars1, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Accuracy (Overall)
bars2 = ax2.bar(models, accuracies, color=accuracy_color, alpha=0.8, edgecolor='white', linewidth=1)
ax2.set_title('Accuracy (Overall)', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_ylim(0.88, 1.0)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, value in zip(bars2, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Styling
for ax in [ax1, ax2]:
    ax.set_xlabel('Models', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)

# Add main title
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# Print summary table
print("Model Performance Summary (Tuned Models):")
print("=" * 45)
print(f"{'Model':<6} {'F1-Score':<10} {'Accuracy':<10}")
print("-" * 45)
for model, f1, acc in zip(models, f1_scores, accuracies):
    print(f"{model:<6} {f1:<10.3f} {acc:<10.3f}")

print(f"\nBest F1-Score: {models[np.argmax(f1_scores)]} ({max(f1_scores):.3f})")
print(f"Best Accuracy: {models[np.argmax(accuracies)]} ({max(accuracies):.3f})")