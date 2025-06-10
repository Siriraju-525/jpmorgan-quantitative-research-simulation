import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
file_name = "credit_risk_dataset.csv"
df = pd.read_csv(file_name)

# Display first few rows
print("Dataset Preview:\n", df.head())

# Identify the target column
target_column = "loan_status"  # Change based on actual column name in CSV

# Check if the target column exists
if target_column not in df.columns:
    raise ValueError(f"Error: Column '{target_column}' not found in dataset!")
# Separate features and target variable
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical variables to numbers
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Handle missing values
X.fillna(X.median(), inplace=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Predict probability of default for a new loan
# Ensure new_loan has the same number of features as X
new_loan = np.array([[22, 59000, 1, 123, 1, 35000, 16.02, 1, 0.59, 2, 3]])  # Modify based on dataset

# Scale the new loan data
new_loan_scaled = scaler.transform(new_loan)

# Predict probability of default
probability_of_default = model.predict_proba(new_loan_scaled)[:, 1][0]

# Calculate expected loss (assuming 10% recovery rate)
recovery_rate = 0.1
expected_loss = probability_of_default * (1 - recovery_rate)

print(f"✅ Probability of Default: {probability_of_default:.2f}")
print(f"✅ Expected Loss on Loan: ${expected_loss:.2f}")