import pandas as pd

# Read the log_likelihoods.csv file
df = pd.read_csv('log_likelihoods.csv')

# Process the data as needed
# For example, let's assume we want to split the data based on a column 'category'
Files = df['File'].unique()


import matplotlib.pyplot as plt

# Create a histogram for each category
plt.figure(figsize=(10, 6))
df['Log Likelihood'].hist(bins=30)
plt.title('Histogram of Log Likelihoods')
plt.xlabel('Log Likelihood')
plt.ylabel('Frequency')
plt.savefig('log_likelihoods_histogram.png')

print("Histogram of Log Likelihoods has been created and saved as log_likelihoods_histogram.png.")