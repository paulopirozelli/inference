import random
import string
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertForBinaryClassification(nn.Module):
    def __init__(self, vocab_size, num_hidden_layers, num_attention_heads, 
                 dropout_prob, hidden_size, max_position_embeddings, model_type):
        super(BertForBinaryClassification, self).__init__()
        
        # Store model_type as an instance attribute
        self.model_type = model_type

        # Load pre-trained BERT model or use custom configuration
        configuration = BertConfig()
        configuration.vocab_size = vocab_size
        configuration.num_hidden_layers = num_hidden_layers
        configuration.num_attention_heads = num_attention_heads
        configuration.hidden_dropout_prob = dropout_prob
        configuration.hidden_size = hidden_size
        configuration.max_position_embeddings = max_position_embeddings
        
        self.bert = BertModel(configuration)
        
        # Classification head
        if self.model_type == 'bert':
            self.decoder = nn.Linear(hidden_size, 1, bias=False)
        elif self.model_type == 'bertlogit':
            self.decoder_initial = nn.Linear(hidden_size, max_position_embeddings, bias=False)
            self.decoder_final = nn.Linear(max_position_embeddings, 1, bias=False)
            self.sigmoid = nn.Sigmoid()
        elif self.model_type == 'bertmean':
            self.decoder_initial = nn.Linear(hidden_size, max_position_embeddings, bias=False)
            self.decoder_final = nn.Linear(max_position_embeddings, 1, bias=False)
            # Initialize decoder_final weights to 1 / max_position_embeddings and make it non-trainable
            with torch.no_grad():
                self.decoder_final.weight.fill_(1.0 / max_position_embeddings)
            self.decoder_final.weight.requires_grad = False
            self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        # Get outputs from BERT model
        outputs = self.bert(input_ids=input_ids, output_hidden_states=True, output_attentions=True)

        # Extract hidden states and attention maps
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # Use the last hidden state for classification
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Apply the appropriate classification layer based on model_type
        if self.model_type == 'bert':
            logits = self.decoder(last_hidden_state)
            probs = torch.sigmoid(logits)  # Binary classification
        elif self.model_type in ['bertlogit', 'bertmean']:
            logits_first = self.decoder_initial(last_hidden_state)
            logits_final = self.decoder_final(logits_first)
            probs = self.sigmoid(logits_final)

        return probs, hidden_states, attentions

def create_argument(inference_level: int):
    predicates = list(string.ascii_lowercase)

    fwd_rules, initial_fact, consequent = forward_rules(inference_level, predicates)

    set_rules, label = get_rules(fwd_rules, inference_level)

    # Shuffle the order of rules and facts
    random.shuffle(set_rules)

    # Concatenate in a string
    #rules = 'R'.join(set_rules)
    rules = ''.join(set_rules)

    # Merge rules, facts, and hypothesis
    text = initial_fact + rules + consequent
    #text = 'S' + initial_fact + 'R' + rules + 'E' + consequent + 'L'

    return text, label

def number_rules(inference_level: int):
    '''
    Return the number of rules in the argument
    '''
    return 3 * (inference_level + 1)

def forward_rules(inference_level: int, predicates: list):
    '''
    Generate the forward rules of the argument
    '''

    remaining_facts = predicates
    fwd_rules = []

    arg_number = number_rules(inference_level)

    for level in range(arg_number-1):
        if level == 0:
            initial_fact = random.choice(predicates)[0]
            remaining_facts.remove(initial_fact)

            first_conseq = False

            while first_conseq == False:
                consequent = random.choice(predicates)[0]
                if consequent == initial_fact:
                    first_conseq = False
                else:
                    remaining_facts.remove(consequent)
                    rule = initial_fact + consequent
                    fwd_rules.append(rule)
                    first_conseq = True

        else:
            antecedent = fwd_rules[-1][-1]

            next_conseq = False

            while next_conseq == False:
                consequent = random.choice(remaining_facts)[0]
                if consequent == antecedent:
                    next_conseq = False
                else:
                    remaining_facts.remove(consequent)
                    rule = antecedent + consequent
                    fwd_rules.append(rule)
                    next_conseq = True

    return fwd_rules, initial_fact, consequent

def get_rules(fwd_rules: list, inference_level: int):
    '''
    Break the full argument and create True and False instances
    '''
    set_rules = fwd_rules

    if get_variable_with_probability(0.5):
        set_rules.pop(inference_level - 1)
        set_rules.pop(-inference_level)
        label = 'F'
    else:
        set_rules.pop(inference_level)
        set_rules.pop(-(inference_level + 1))
        label = 'T'

    return set_rules, label

def get_variable_with_probability(probability: float):
    return random.random() < probability

def get_index(vocab, word):
    return vocab[word]

def tokenized_dataloader(df, vocab, batch_size):
    df['input_ids'] = df['text'].apply(lambda x: [get_index(vocab, word) for word in x])

    df['labels'] = df['labels'].map({'T': 1, 'F': 0})

    from datasets import Dataset
    from torch.utils.data import DataLoader
    ds = Dataset.from_dict({"input_ids": df['input_ids'], "labels": df['labels']}).with_format("torch")

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader

# Function to plot attention maps and save them as images
def plot_attention(attentions, tokens, layer, head, input_text, label, prediction, sentence_idx=0, save_dir='attention_maps'):
    """
    Plot attention map for a specific layer and head, and save it as an image.
    Add input sentence, label, and prediction to the bottom of the figure.
    """
    # Extract attention for a specific layer and head
    attention = attentions[layer][sentence_idx, head].detach().cpu().numpy()  # Shape: (seq_len, seq_len)
    
    # Create the directory to save attention maps if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot the attention map
    plt.figure(figsize=(8, 8))
    plt.imshow(attention, cmap='viridis')
    plt.colorbar()
    plt.title(f'Layer {layer + 1}, Head {head + 1}', fontsize=16)
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=12)
    plt.yticks(range(len(tokens)), tokens, fontsize=12)

    # Add the input, label, and prediction as text at the bottom of the plot
    plt.figtext(0.5, -0.1, f'Input: {input_text}', wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.5, -0.15, f'Label: {label}, Prediction: {prediction}', wrap=True, horizontalalignment='center', fontsize=12)

    # Save the figure with a descriptive filename
    save_path = os.path.join(save_dir, f'attention_layer{layer+1}_head{head+1}.png')
    plt.savefig(save_path, bbox_inches='tight')

# Function to pass through a single observation and plot attentions
def analyze_attention_single_batch(dataloader, model, vocab):
    model.eval()
    
    for batch in dataloader:
        input_ids = batch['input_ids']  # Assuming batch size 1
        labels = batch['labels']  # Extract the labels from the batch

        # Forward pass through the model
        with torch.no_grad():
            predictions, _, attentions = model(batch['input_ids'])  # Get predictions and attentions
        
        # Assuming input_ids contains token indices, map them to actual tokens
        reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse the vocab so that the keys are the indices and the values are the tokens
        tokens = [reverse_vocab[idx.item()] for idx in input_ids[0]]  # Map each tensor element directly to token

        # Convert tokens to a string to represent the input text
        input_text = ''.join(tokens)  # Join the tokens into a single string

        # Extract label and prediction
        label = labels[0].item()  # Assuming batch size 1, extract the label
        prediction = round(predictions[0].item())  # Assuming batch size 1, round the prediction to 0 or 1
        
        # Plot attentions for each layer and head
        num_layers = len(attentions)
        num_heads = attentions[0].size(1)

        for layer in range(num_layers):
            for head in range(num_heads):
                plot_attention(attentions, tokens, layer, head, input_text, label, prediction)

        break  # We only want to process a single batch
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


# train and test steps
def train_step(model, split_size, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()

    total_loss = 0

    predictions = []
    correct_labels = []

    for batch in data_loader:
        # Send data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 1. Forward pass
        logits, _, _ = model(batch['input_ids'])

        #2. Calculate loss
        last_tokens = logits.squeeze() # Select first token of each sequence

        loss = loss_fn(last_tokens, batch['labels'].float()) 
        total_loss += loss.item() * batch['labels'].size(0)

        # Get accuracy
        y_pred = (last_tokens >= 0.5).float()  # get predicted class
        y_label = batch['labels']
        predictions.append(y_pred)
        correct_labels.append(y_label)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        scheduler.step()

    # Concatenate tensors
    predictions = torch.cat(predictions, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)

    # Calculate loss and accuracy per epoch and print out what's happening
    total_loss = total_loss / split_size
    mean_acc = accuracy_fn(predictions, correct_labels)
    
    return total_loss, mean_acc

def validation_step(model, split_size, data_loader, loss_fn, device):
    model.eval()  # put model in eval mode

    total_loss = 0

    predictions = []
    correct_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        for batch in data_loader:
            # Send data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1. Forward pass
            logits, _, _ = model(batch['input_ids'])

            #2. Calculate loss
            last_tokens = logits.squeeze() # Select last token of each sequence
                
            loss = loss_fn(last_tokens, batch['labels'].float()) 
            total_loss += loss.item() * batch['labels'].size(0)

            # Get accuracy
            y_pred = (last_tokens >= 0.5).float()  # get predicted class
            y_label = batch['labels']
            predictions.append(y_pred)
            correct_labels.append(y_label)

        # Concatenate tensors
        predictions = torch.cat(predictions, dim=0)
        correct_labels = torch.cat(correct_labels, dim=0)

        # Calculate loss and accuracy per epoch and print out what's happening
        total_loss = total_loss / split_size
        mean_acc = accuracy_fn(predictions, correct_labels)
    
        return total_loss, mean_acc

def plot_attention_grid(attentions, tokens, input_text, label_str, prediction, inference_level, num_layers, num_attention_heads):
    # Same function body, but now with inference_level, num_layers, and num_attention_heads passed as arguments
    num_heads = attentions[0].size(1)
    
    # Set up a figure with a grid of subplots
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))  # Customize figure size based on layers and heads
    
    # Loop over layers and heads to plot each attention map
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]  # Select the subplot corresponding to this layer and head
            attention = attentions[layer][0, head].detach().cpu().numpy()  # Get the attention map for the current layer and head
            cax = ax.matshow(attention, cmap='viridis')  # Plot attention as a heatmap
            ax.set_title(f"Layer {layer+1}, Head {head+1}")
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
    
    fig.subplots_adjust(right=0.85)  # Adjust the subplot grid to make room for the color bar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position for the color bar
    fig.colorbar(cax, cax=cbar_ax)  # Add the color bar to the figure
    
    # Set the overall title for the figure
    fig.suptitle(f'''Attention Visualization for Input: '{input_text}'\nLabel: {label_str}\n BERT Model for Inference Level {inference_level}, {num_layers} Layers, {num_attention_heads} Attention Heads.''', y=1.05)
    
    # Save the figure with a descriptive filename
    save_path = os.path.join('attention_maps', f'attention_map_{label_str}_{inference_level}_{num_layers}_{num_attention_heads}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def analyze_single_observation(observation, model, vocab, label_str, inference_level, num_layers, num_attention_heads):
    input_ids = observation['input_ids'].unsqueeze(0)  # Add batch dimension
    labels = observation['labels'].unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        predictions, _, attentions = model(input_ids)  # Get predictions and attentions

    # Assuming input_ids contains token indices, map them to actual tokens
    reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse the vocab so that the keys are the indices and the values are the tokens
    tokens = [reverse_vocab[idx.item()] for idx in input_ids[0]]  # Map each tensor element directly to token

    # Convert tokens to a string to represent the input text
    input_text = ''.join(tokens)  # Join the tokens into a single string

    # Extract label and prediction
    prediction = round(predictions[0].item())  # Assuming batch size 1, round the prediction to 0 or 1

    # Call the plot function and pass the label_str ('True' or 'False')
    plot_attention_grid(attentions, tokens, input_text, label_str, prediction, inference_level, num_layers, num_attention_heads)

def analyze_attention_single_batch(dataloader, model, vocab, inference_level, num_layers, num_attention_heads):
    model.eval()

    true_batches = []
    false_batches = []
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        for i, label in enumerate(labels):
            if label.item() == 1:
                true_batches.append({'input_ids': input_ids[i], 'labels': labels[i]})
            else:
                false_batches.append({'input_ids': input_ids[i], 'labels': labels[i]})

    if len(true_batches) > 0:
        random_true = random.choice(true_batches)
        analyze_single_observation(random_true, model, vocab, 'True', inference_level, num_layers, num_attention_heads)

    if len(false_batches) > 0:
        random_false = random.choice(false_batches)
        analyze_single_observation(random_false, model, vocab, 'False', inference_level, num_layers, num_attention_heads)