import pandas as pd
import json

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Get 2 fraud samples and 2 legitimate samples
fraud_samples = df[df['Class'] == 1].head(2)
legit_samples = df[df['Class'] == 0].head(2)

# Combine and convert to dict
samples = pd.concat([fraud_samples, legit_samples])
samples = samples.drop('Class', axis=1)

# Save as JSON
samples_dict = samples.to_dict(orient='records')

# Print for copying
for i, sample in enumerate(samples_dict):
    print(f"Sample {i+1}:")
    print(json.dumps(sample, indent=2))
    print("\n")