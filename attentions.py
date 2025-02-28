import pandas as pd
import string
import sys
import os
import torch
from sklearn.model_selection import train_test_split
from models import BertForBinaryClassification
from utils import tokenized_dataloader, analyze_attention_single_batch, validation_step
import torch.nn as nn

# Hyperparameters
folder_name = sys.argv[1]
model_name = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split the model name into parts
model_parts = model_name.split('/')[-1].split('_')

# Parse the fixed parts of the model name
inference_level = int(model_parts[0])
model_type = model_parts[1]
num_layers = int(model_parts[2])
num_attention_heads = int(model_parts[3])
hidden_size = int(model_parts[4])
dropout_prob = float(model_parts[5])
learning_rate = float(model_parts[6])
batch_size = int(model_parts[7])
num_epochs = int(model_parts[8])
convergence = float(model_parts[9])

# Handle filtered (True/False) and letter (optional parts)
filtered = 'filtered' if model_parts[10] == 'True' else None
letter = model_parts[11][0] if filtered else 'S'

# Import dataframe
dataset_name = 'data_' + str(inference_level) + '.csv'
data = pd.read_csv(dataset_name)

# Split dataset
df_train, df_validation = train_test_split(data, train_size=0.9, random_state=42, stratify=data['labels'])
df_validation, df_test = train_test_split(df_validation, test_size=0.5, random_state=42, stratify=df_validation['labels'])

train_size, validation_size, test_size = len(df_train), len(df_validation), len(df_test)

# Add CLS token
df_train['text'] = df_train['text'].apply(lambda x: 'S' + x)
df_validation['text'] = df_validation['text'].apply(lambda x: 'S' + x)
df_test['text'] = df_test['text'].apply(lambda x: 'S' + x)

# Context size
max_position_embeddings = len(df_test.iloc[0,0]) # the maximum context length for predictions

# Create vocabulary
predicates = ['S'] + list(string.ascii_lowercase)
indexes = [i for i, _ in enumerate(predicates)]

vocab = dict(zip(predicates, indexes))

# Get vocab size
vocab_size = len(vocab)

# Tokenize sentences and create Dataloader
test_dataloader = tokenized_dataloader(df_test, vocab, batch_size)

# Create model
model = BertForBinaryClassification(vocab_size=vocab_size, num_hidden_layers=num_layers, 
            num_attention_heads=num_attention_heads, dropout_prob=dropout_prob,
            hidden_size=hidden_size, max_position_embeddings=max_position_embeddings,
            model_type=model_type).to(device)

# Load model
folder = folder_name

full_name = os.path.join(folder, model_name)

model.load_state_dict(torch.load(full_name, map_location=torch.device(device)))

analyze_attention_single_batch(test_dataloader, model, vocab, inference_level, num_layers, num_attention_heads, num_epochs, device)