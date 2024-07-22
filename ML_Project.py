import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
data_path = 'path_to_your_csv/purchase_history.csv'
df = pd.read_csv('purchase_history.csv')

# Preprocess the data
X = df[['Age', 'Salary', 'Price']]
y = df['Purchased']

# Train a Decision Tree Classifier for feature selection
dt_feature_selector = DecisionTreeClassifier(random_state=42)
dt_feature_selector.fit(X, y)
importance_scores = pd.Series(dt_feature_selector.feature_importances_, index=X.columns)
threshold = 0.1
important_features = importance_scores[importance_scores > threshold]

# Streamlit app
st.title('Purchase Behavior Prediction')

# Plot feature importance
fig, ax = plt.subplots(figsize=(7, 7))
sns.barplot(x=importance_scores.values, y=importance_scores.index, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)

# Add a horizontal line
st.write("---")  # This creates a horizontal line

# Plot feature importance
fig, ax = plt.subplots(figsize=(3,3))
plt.pie(important_features, labels=important_features.index, autopct='%1.0f%%', startangle=140)
plt.title('Feature Importance')
st.pyplot(fig)

# Add a horizontal line
st.write("---")  # This creates a horizontal line

# Feature selection
X_selected = X[important_features.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train classifiers
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(X_scaled, y)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_scaled, y)

# User input
st.write("### Enter values for prediction")
user_input = {}
for feature in important_features.index:
    user_input[feature] = st.number_input(f"Enter {feature}", min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))

user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(user_df)

# Predictions
svm_prediction = svm_classifier.predict(user_scaled)[0]
knn_prediction = knn_classifier.predict(user_scaled)[0]

st.write(f"### SVM Prediction: {svm_prediction}")
st.write(f"### KNN Prediction: {knn_prediction}")

# Add a horizontal line
st.write("---")  # This creates a horizontal line

# Calculate metrics
y_pred_svm = svm_classifier.predict(X_scaled)
y_pred_knn = knn_classifier.predict(X_scaled)

accuracy_svm = accuracy_score(y, y_pred_svm)
accuracy_knn = accuracy_score(y, y_pred_knn)

precision_svm = precision_score(y, y_pred_svm, average='binary')
precision_knn = precision_score(y, y_pred_knn, average='binary')

recall_svm = recall_score(y, y_pred_svm, average='binary')
recall_knn = recall_score(y, y_pred_knn, average='binary')

f1_svm = f1_score(y, y_pred_svm, average='binary')
f1_knn = f1_score(y, y_pred_knn, average='binary')

st.write("### Model Performance Metrics")

st.write("#### SVM Metrics")
st.write(f"Accuracy: {accuracy_svm:.2f}")
st.write(f"Precision: {precision_svm:.2f}")
st.write(f"Recall: {recall_svm:.2f}")
st.write(f"F1 Score: {f1_svm:.2f}")

st.write("#### KNN Metrics")
st.write(f"Accuracy: {accuracy_knn:.2f}")
st.write(f"Precision: {precision_knn:.2f}")
st.write(f"Recall: {recall_knn:.2f}")
st.write(f"F1 Score: {f1_knn:.2f}")

# Add a horizontal line
st.write("---")  # This creates a horizontal line

# Confusion Matrix
cm_svm = confusion_matrix(y, y_pred_svm)
cm_knn = confusion_matrix(y, y_pred_knn)

# Plot confusion matrix using seaborn
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix - SVM')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix - KNN')
st.pyplot(fig)

# Add a horizontal line
st.write("---")  # This creates a horizontal line

# Determine and display the preferred classifier
preferred_classifier = 'SVM' if svm_prediction > knn_prediction else 'KNN' if svm_prediction < knn_prediction else 'Both (Equal Predictions)'
st.write(f"### Preferred Classifier based on Predictions: {preferred_classifier}")
