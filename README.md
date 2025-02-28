# MATS Project  

This repository contains the code for the project conducted as part of the application process for **MATS 8.0**.  

## Repository Structure  

This repository includes four main scripts:  

### 1. `data_generation.py`  

This script generates synthetic data and saves it as a CSV file. It has two hyperparameters:  
- **Inference Level** – The level of logical inference in the dataset.  
- **Number of Parameters** – The total number of data points to generate.  

### 2. `training.py`  

This script trains a customized Transformer model on the logical reasoning dataset.  

#### Parameters:  
- `inference_level`: Integer, defines the level of logical inference.  
- `model_type`: String, specifies the type of model.  
- `num_layers`: Integer, number of layers in the model.  
- `num_attention_heads`: Integer, number of attention heads.  
- `hidden_size`: Integer, hidden layer size.  
- `dropout_prob`: Float, dropout probability.  
- `learning_rate`: Float, learning rate for optimization.  
- `batch_size`: Integer, batch size for training.  
- `num_epochs`: Integer, number of training epochs.  
- `convergence`: Float, convergence threshold.  

### 3. `attentions.py`  

This script generates attention maps for a trained model using random inputs.  

#### Parameters:  
- `folder_name`: String, specifies the folder containing model checkpoints.  
- `model_name`: String, name of the model to analyze.  

### 4. `util_functions.py`  

This script contains all the utility functions used across the other scripts, such as:  
- Data preprocessing functions  
- Model initialization and configuration utilities  
- Training and evaluation helpers  
- Attention visualization tools  
