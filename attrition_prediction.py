import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\elakk\OneDrive\Music\Desktop\Hr analytics\dataset\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert Attrition column
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

# Features
X = df[['Age','MonthlyIncome','YearsAtCompany','JobSatisfaction']]

# Target
y = df['Attrition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict attrition
df['Predicted_Attrition'] = model.predict(X)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save updated dataset
df.to_csv(r"C:\Users\elakk\OneDrive\Music\Desktop\Hr analytics\dataset\HR_Analytics_Predicted.csv", index=False)

print("Prediction file saved successfully!")