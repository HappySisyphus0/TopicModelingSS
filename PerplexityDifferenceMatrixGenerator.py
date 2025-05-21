import os
import pandas as pd
import numpy as np

data_folder = 'LogLikelihoodData'

file_patterns = [
    'log_likelihoods_good_prompt_97_1.csv',
    'log_likelihoods_good_prompt_97_2.csv',
    'log_likelihoods_good_prompt_97_3.csv',
    'log_likelihoods_good_prompt_97_4.csv',
    'log_likelihoods_good_prompt_97_5.csv',
    'log_likelihoods_no_prompt_97.csv'
]

# Simplify file names for column naming
simplified_names = {
    'log_likelihoods_good_prompt_97_1.csv': 'p1',
    'log_likelihoods_good_prompt_97_2.csv': 'p2',
    'log_likelihoods_good_prompt_97_3.csv': 'p3',
    'log_likelihoods_good_prompt_97_4.csv': 'p4',
    'log_likelihoods_good_prompt_97_5.csv': 'p5',
    'log_likelihoods_no_prompt_97.csv': 'np'
}

# Dictionary to store perplexity data for each file
perplexity_data = {}
file_columns = {}

# Read the Perplexity column and File column from each file, filtering out rows with # of Tokens < 100
for pattern in file_patterns:
    file_path = os.path.join(data_folder, pattern)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data[data['# of Tokens'] >= 100]  # Filter rows where # of Tokens >= 100
        perplexity_data[simplified_names[pattern]] = data['Perplexity'].values
        file_columns[simplified_names[pattern]] = data['File'].values
    else:
        print(f"File not found: {file_path}")

# Create a DataFrame to store differences
index_length = min(len(perplexity_data[file]) for file in perplexity_data)  # Use the shortest file length
difference_df = pd.DataFrame(index=range(index_length))

# Add the file column from the first file as the index
difference_df['File'] = file_columns[list(simplified_names.values())[0]][:index_length]

# Calculate absolute differences and add them as columns (avoid repeats)
file_names = list(simplified_names.values())
for i, file1 in enumerate(file_names):
    for j in range(i + 1, len(file_names)):  # Start inner loop from i + 1 to avoid repeats
        file2 = file_names[j]
        
        # Ensure both files have the same number of rows
        min_length = min(len(perplexity_data[file1]), len(perplexity_data[file2]))
        perplexity1 = perplexity_data[file1][:min_length]
        perplexity2 = perplexity_data[file2][:min_length]
        
        # Calculate absolute differences
        column_name = f"{file1}-{file2}"
        difference_df[column_name] = perplexity1 - perplexity2

# Write the differences DataFrame to a CSV file
output_file = 'perplexity_differences_matrix.csv'
difference_df.to_csv(output_file, index=False)

print(f"Perplexity differences matrix written to {output_file}")

# Calculate population statistics for each file
stats = []
for file, perplexity in perplexity_data.items():
    stats.append({
        'File': file,
        'Mean': np.mean(perplexity),
        'Median': np.median(perplexity),
        'Variance': np.var(perplexity),
        'Standard Deviation': np.std(perplexity),
        'Min': np.min(perplexity),
        'Max': np.max(perplexity)
    })

# Write population statistics to a separate CSV file
stats_df = pd.DataFrame(stats)
stats_output_file = 'perplexity_population_stats.csv'
stats_df.to_csv(stats_output_file, index=False)

print(f"Population statistics written to {stats_output_file}")

# Calculate statistics for differences
difference_stats = []
for column in difference_df.columns:
    if column != 'File':  # Skip the 'File' column
        difference_stats.append({
            'Difference Pair': column,
            'Mean': difference_df[column].mean(),
            'Median': difference_df[column].median(),
            'Variance': difference_df[column].var(),
            'Standard Deviation': difference_df[column].std(),
            'Min': difference_df[column].min(),
            'Max': difference_df[column].max()
        })

# Write difference statistics to a separate CSV file
difference_stats_df = pd.DataFrame(difference_stats)
difference_stats_output_file = 'perplexity_difference_stats.csv'
difference_stats_df.to_csv(difference_stats_output_file, index=False)

print(f"Difference statistics written to {difference_stats_output_file}")