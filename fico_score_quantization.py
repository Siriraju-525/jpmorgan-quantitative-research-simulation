import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("fico_credit_data.csv")

# Define the number of buckets
num_buckets = 5  

# Create bucket boundaries (dividing FICO scores into equal intervals)
df['bucket'] = pd.qcut(df['fico_score'], num_buckets, labels=False)

# Compute probability of default per bucket
bucket_data = df.groupby('bucket').agg(
    total_customers=('default', 'count'),
    defaults=('default', 'sum')
).reset_index()

bucket_data['p_default'] = bucket_data['defaults'] / bucket_data['total_customers']

# Function to calculate Mean Squared Error (MSE)
def mse_loss(bucket_data):
    avg_default = df['default'].mean()  
    return mean_squared_error(bucket_data['p_default'], [avg_default] * len(bucket_data))
# Function to calculate Log-Likelihood
def log_likelihood(bucket_data):
    p = bucket_data['p_default'].clip(1e-5, 1 - 1e-5)  # Avoid log(0)
    return -np.sum(bucket_data['defaults'] * np.log(p) + (bucket_data['total_customers'] - bucket_data['defaults']) * np.log(1 - p))

# Optimize bucket boundaries using MSE
result_mse = minimize(lambda x: mse_loss(bucket_data), x0=[0.5] * num_buckets, method='Nelder-Mead')
optimal_mse = result_mse.fun

# Optimize bucket boundaries using Log-Likelihood
result_ll = minimize(lambda x: log_likelihood(bucket_data), x0=[0.5] * num_buckets, method='Nelder-Mead')
optimal_ll = result_ll.fun

# Display Results
print(f"Mean Squared Error: {optimal_mse:.4f}")
print(f"Log-Likelihood: {optimal_ll:.4f}")
print("\nBucket-wise Default Probabilities:")
print(bucket_data)