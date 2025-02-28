import pandas as pd
import string
import csv
import sys
import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import get_constant_schedule_with_warmup
from models import *
import torch.nn as nn
from utils import tokenized_dataloader, train_step, validation_step, BertForBinaryClassification

# Hyperparameters
inference_level = int(sys.argv[1])
model_type = sys.argv[2]
num_layers = int(sys.argv[3])
num_attention_heads = int(sys.argv[4])
hidden_size = int(sys.argv[5])
dropout_prob = float(sys.argv[6])
learning_rate = float(sys.argv[7])
batch_size = int(sys.argv[8])
num_epochs = int(sys.argv[9])
convergence = float(sys.argv[10])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import dataframe
dataset_name = 'data_' + str(inference_level) + '.csv'
data = pd.read_csv(dataset_name)

# Split dataset
df_train, df_validation = train_test_split(data, train_size=0.9, random_state=42, stratify=data['labels'])
df_validation, df_test = train_test_split(df_validation, test_size=0.5, random_state=42, stratify=df_validation['labels'])

train_size, validation_size, test_size = len(df_train), len(df_validation), len(df_test)

# Print dataset size
print(f'Train size: {train_size} | Validation size: {validation_size} | Test size: {test_size}')

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
train_dataloader = tokenized_dataloader(df_train, vocab, batch_size)
validation_dataloader = tokenized_dataloader(df_validation, vocab, batch_size)
test_dataloader = tokenized_dataloader(df_test, vocab, batch_size)

# Create model
model = BertForBinaryClassification(vocab_size=vocab_size, num_hidden_layers=num_layers, 
    num_attention_heads=num_attention_heads, dropout_prob=dropout_prob,
    hidden_size=hidden_size, max_position_embeddings=max_position_embeddings,
    model_type=model_type).to(device)

# Define the primary loss function
loss_fn = nn.BCELoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define scheduler
num_warmup_steps = len(train_dataloader)
num_training_steps = num_epochs*len(train_dataloader)

#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

# Training
train_results_loss = []
train_results_acc = []
validation_results_loss = []
validation_results_acc = []

for epoch in range(1, num_epochs+1):

    print(f"---------\nEpoch: {epoch}")

    # Train the model
    train_loss, train_acc = train_step(model=model,
        split_size=train_size,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    train_results_loss.append(train_loss)
    train_results_acc.append(train_acc)

    print(f"Train loss: {train_loss} | Train accuracy: {train_acc:.2f}%")

    # Validate the model 
    validation_loss, validation_acc = validation_step(model=model,
        split_size=validation_size,
        data_loader=validation_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    # Save results
    validation_results_loss.append(validation_loss)
    validation_results_acc.append(validation_acc)

    print(f"Validation loss: {validation_loss} | Validation accuracy: {validation_acc:.2f}%")

    if validation_acc > convergence:
        print('Convergence achieved during training.')

        # Create the 'Models' folder if it doesn't exist
        folder_model = 'Models'
        os.makedirs(folder_model, exist_ok=True)

        # Save model
        full_name = str(inference_level) + '_' + model_type + '_' + str(num_layers) + '_' + \
            str(num_attention_heads) + '_' + str(hidden_size) + '_' + str(dropout_prob) + '_' + \
            str(learning_rate) + '_' + str(batch_size) + '_' + str(epoch) + '_' + \
            str(convergence) + '.pth'
        
        file_name = os.path.join(folder_model, full_name)

        torch.save(model.state_dict(), file_name)

        # Test final model
        test_loss, test_acc = validation_step(model=model,
        split_size=test_size,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        device=device
        )

        print(f'Test loss: {test_loss} | Test accuracy: {test_acc:.2f}%')

        break

else:
    print('Convergence not achieved during training.')

    # Test final model
    test_loss, test_acc = validation_step(model=model,
        split_size=test_size,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    print(f'Test loss: {test_loss} | Test accuracy: {test_acc:.2f}%')


