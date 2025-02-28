import pandas as pd
import string
import sys
from utils import create_argument

# Hyperparameters
inference_level = int(sys.argv[1])
sample_number = int(sys.argv[2])

print(f'Creating dataset. Inference level: {inference_level}. Size: {sample_number}')

predicates = list(string.ascii_lowercase)

lst = []

for i in range(int(sample_number*1.1)):
    text, label = create_argument(inference_level)
    lst.append([text, label])

df = pd.DataFrame(lst, columns=['text', 'labels'])

# Eliminate duplicate observations
df = df.drop_duplicates(subset=['text'])

# Balance dataset
# Group the dataframe by the label column
grouped = df.groupby('labels')

# Create an empty dataframe to store the sampled observations
sampled_df = pd.DataFrame()

# Sample n observations from each group
for _, group_df in grouped:
    sampled_df = pd.concat([sampled_df, group_df.sample(int(sample_number/2))])

# Reset the index of the sampled dataframe
df = sampled_df.reset_index(drop=True)

# Check label distribution
print(f'Sample for inference level {inference_level}:', '\n', df.head(), 
      '\n', 'Label distribution:', '\n', df['labels'].value_counts())

# Define name
data_name = 'data_' + str(inference_level) + '.csv'

# Save to csv
df.to_csv(data_name, index=False)
