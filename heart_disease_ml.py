# ============================================================
#   HEART DISEASE PREDICTION - MACHINE LEARNING PROJECT
#   Course: Pharmaceutical Intelligence with Data Analytics
# ============================================================

# -------------------------------------------------------
# STEP 1: IMPORT ALL LIBRARIES
# -------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)

print("=" * 60)
print("  HEART DISEASE PREDICTION - ML PROJECT")
print("  Pharmaceutical Intelligence with Data Analytics")
print("=" * 60)

# -------------------------------------------------------
# STEP 2: LOAD THE DATASET
# -------------------------------------------------------
print("\n[STEP 1] Loading Dataset...")

df = pd.read_excel("heart_disease_dataset.xlsx", sheet_name="Heart Disease Data")

print(f"  Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n  First 5 rows of data:")
print(df.head())

# -------------------------------------------------------
# STEP 3: EXPLORE THE DATA
# -------------------------------------------------------
print("\n[STEP 2] Exploring the Data...")
print("\n  Dataset Info:")
print(df.info())

print("\n  Basic Statistics:")
print(df.describe().round(2))

print(f"\n  Missing Values: {df.isnull().sum().sum()} (Total)")
print(f"  Heart Disease Cases:    {df['target'].sum()} patients (target = 1)")
print(f"  No Heart Disease Cases: {(df['target'] == 0).sum()} patients (target = 0)")

# -------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA) - VISUALIZATIONS
# -------------------------------------------------------
print("\n[STEP 3] Creating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Heart Disease - Exploratory Data Analysis (EDA)",
             fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Target distribution (Pie Chart)
ax1 = axes[0, 0]
target_counts = df['target'].value_counts()
colors = ['#E74C3C', '#2ECC71']
ax1.pie(target_counts, labels=['Heart Disease', 'No Heart Disease'],
        autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'fontsize': 11})
ax1.set_title('Distribution of Heart Disease Cases', fontweight='bold')

# Plot 2: Age distribution by target
ax2 = axes[0, 1]
df[df['target'] == 1]['age'].hist(ax=ax2, alpha=0.7, color='#E74C3C',
                                   label='Heart Disease', bins=15)
df[df['target'] == 0]['age'].hist(ax=ax2, alpha=0.7, color='#2ECC71',
                                   label='No Heart Disease', bins=15)
ax2.set_title('Age Distribution by Disease Status', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Number of Patients')
ax2.legend()

# Plot 3: Sex vs Heart Disease
ax3 = axes[0, 2]
sex_disease = df.groupby(['sex', 'target']).size().unstack()
sex_disease.index = ['Female', 'Male']
sex_disease.columns = ['No Disease', 'Disease']
sex_disease.plot(kind='bar', ax=ax3, color=['#2ECC71', '#E74C3C'],
                 edgecolor='white', width=0.6)
ax3.set_title('Heart Disease by Sex', fontweight='bold')
ax3.set_xlabel('Sex')
ax3.set_ylabel('Number of Patients')
ax3.set_xticklabels(['Female', 'Male'], rotation=0)
ax3.legend()

# Plot 4: Cholesterol vs Age (Scatter)
ax4 = axes[1, 0]
colors_scatter = df['target'].map({1: '#E74C3C', 0: '#2ECC71'})
ax4.scatter(df['age'], df['chol'], c=colors_scatter, alpha=0.6, s=40)
ax4.set_title('Cholesterol vs Age', fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Cholesterol (mg/dl)')
red_patch = mpatches.Patch(color='#E74C3C', label='Heart Disease')
green_patch = mpatches.Patch(color='#2ECC71', label='No Heart Disease')
ax4.legend(handles=[red_patch, green_patch])

# Plot 5: Max Heart Rate by Disease
ax5 = axes[1, 1]
df.boxplot(column='thalach', by='target', ax=ax5,
           boxprops=dict(color='#2C3E50'),
           medianprops=dict(color='#E74C3C', linewidth=2))
ax5.set_title('Max Heart Rate by Disease Status', fontweight='bold')
ax5.set_xlabel('Target (0=No Disease, 1=Disease)')
ax5.set_ylabel('Max Heart Rate (thalach)')
plt.sca(ax5)
plt.title('Max Heart Rate by Disease Status')

# Plot 6: Correlation Heatmap
ax6 = axes[1, 2]
corr = df.corr()
sns.heatmap(corr, ax=ax6, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax6.set_title('Correlation Heatmap', fontweight='bold')
ax6.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('/home/claude/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved successfully!")

# -------------------------------------------------------
# STEP 5: PREPARE DATA FOR MACHINE LEARNING
# -------------------------------------------------------
print("\n[STEP 4] Preparing Data for Machine Learning...")

# Separate features (X) and target (y)
X = df.drop('target', axis=1)  # All columns except target
y = df['target']               # Target column only

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features (normalize values)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples:  {X_test.shape[0]}")
print("  Features scaled successfully!")

# -------------------------------------------------------
# STEP 6: BUILD AND TRAIN THE MODELS
# -------------------------------------------------------
print("\n[STEP 5] Training Machine Learning Models...")

# --- Model 1: Logistic Regression (Baseline) ---
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc  = accuracy_score(y_test, lr_pred)
print(f"  Logistic Regression Accuracy:  {lr_acc * 100:.2f}%")

# --- Model 2: Decision Tree ---
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_acc  = accuracy_score(y_test, dt_pred)
print(f"  Decision Tree Accuracy:        {dt_acc * 100:.2f}%")

# --- Model 3: Random Forest (Best Model) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"  Random Forest Accuracy:        {rf_acc * 100:.2f}%")

# -------------------------------------------------------
# STEP 7: EVALUATE THE MODELS
# -------------------------------------------------------
print("\n[STEP 6] Evaluating Model Performance...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Model Evaluation Results - Heart Disease Prediction",
             fontsize=15, fontweight='bold')

models      = ['Logistic Regression', 'Decision Tree', 'Random Forest']
predictions = [lr_pred, dt_pred, rf_pred]
accuracies  = [lr_acc, dt_acc, rf_acc]
colors_cm   = ['Blues', 'Oranges', 'Reds']

# Top row: Confusion Matrices
for i, (name, pred, cmap) in enumerate(zip(models, predictions, colors_cm)):
    ax = axes[0, i]
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                linewidths=1, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'{name}\nAccuracy: {accuracies[i]*100:.1f}%',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

# Bottom Left: Accuracy Comparison Bar Chart
ax_bar = axes[1, 0]
bar_colors = ['#3498DB', '#F39C12', '#E74C3C']
bars = ax_bar.bar(models, [a * 100 for a in accuracies],
                  color=bar_colors, edgecolor='white', width=0.5)
ax_bar.set_title('Model Accuracy Comparison', fontweight='bold')
ax_bar.set_ylabel('Accuracy (%)')
ax_bar.set_ylim([50, 100])
ax_bar.set_xticklabels(models, rotation=12, ha='right')
for bar, acc in zip(bars, accuracies):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=12)

# Bottom Middle: ROC Curves
ax_roc = axes[1, 1]
roc_colors = ['#3498DB', '#F39C12', '#E74C3C']
for model, name, color in zip([lr_model, dt_model, rf_model], models, roc_colors):
    prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
ax_roc.set_title('ROC Curves - All Models', fontweight='bold')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(fontsize=9)
ax_roc.grid(alpha=0.3)

# Bottom Right: Feature Importance (Random Forest)
ax_fi = axes[1, 2]
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', ax=ax_fi, color='#E74C3C',
                                    edgecolor='white')
ax_fi.set_title('Feature Importance\n(Random Forest)', fontweight='bold')
ax_fi.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('/home/claude/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Evaluation plots saved successfully!")

# -------------------------------------------------------
# STEP 8: DETAILED CLASSIFICATION REPORT
# -------------------------------------------------------
print("\n[STEP 7] Detailed Classification Reports...")
print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_pred,
      target_names=['No Disease', 'Heart Disease']))

print("--- Decision Tree ---")
print(classification_report(y_test, dt_pred,
      target_names=['No Disease', 'Heart Disease']))

print("--- Random Forest (BEST MODEL) ---")
print(classification_report(y_test, rf_pred,
      target_names=['No Disease', 'Heart Disease']))

# -------------------------------------------------------
# STEP 9: FEATURE IMPORTANCE TABLE
# -------------------------------------------------------
print("\n[STEP 8] Top Predictors of Heart Disease (Random Forest):")
importance_df = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

importance_df['Importance %'] = (importance_df['Importance'] * 100).round(2)
print(importance_df[['Feature', 'Importance %']].to_string(index=False))

# -------------------------------------------------------
# STEP 10: FINAL SUMMARY
# -------------------------------------------------------
best_model = models[np.argmax(accuracies)]
best_acc   = max(accuracies)

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Best Model:     {best_model}")
print(f"  Best Accuracy:  {best_acc * 100:.2f}%")
top_feature = importance_df.iloc[0]['Feature']
print(f"  Top Predictor:  {top_feature}")
print("\n  CLINICAL RECOMMENDATION:")
print("  Patients with high 'oldpeak', low 'thalach', and")
print("  abnormal 'thal' values are at highest risk.")
print("  Early screening and pharmaceutical intervention")
print("  is recommended for high-risk patients.")
print("=" * 60)
print("\n  All plots saved:")
print("  - eda_plots.png       (Exploratory Analysis)")
print("  - model_evaluation.png (Model Results)")
print("\n  Project Complete!")
