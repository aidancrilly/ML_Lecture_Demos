"""
Demonstration of Training a Neural Network for Physics Data Analysis using PyTorch.

This script provides a complete workflow for training and evaluating a Multi-Layer 
Perceptron (MLP) neural network using PyTorch. The network is designed 
to predict physical parameters from features of Ion Acoustic Wave (IAW) Thomson scattering
spectra.

This is a practical example of how neural networks can be used as surrogate models 
to quickly approximate the results of complex experiments or simulations.

The script's workflow includes:
1.  **Configuration**: Flags to easily switch between training a new model and loading an 
    existing one.
2.  **Data Handling**: Loading data from a CSV file, splitting it into input features (X) 
    and output targets (Y).
3.  **Preprocessing**: Splitting the data into training and testing sets to evaluate 
    generalization, and standardizing the data (scaling to zero mean and unit variance), 
    which is a crucial step for efficient neural network training.
4.  **Model Architecture**: Defining a sequential PyTorch model with several dense (fully-connected) 
    layers and different activation functions (ReLU, tanh).
5.  **Training**: If training is enabled, the script defines a loss function 
    (Mean Squared Error) and an optimizer (Adam), then fits the model to the training data. 
    The model and its training history are saved.
6.  **Evaluation**: The script plots the training and validation loss over epochs to monitor 
    for overfitting. It then uses the trained model to make predictions on the unseen test 
    data and generates comparison plots to evaluate the model's accuracy.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
# Set these flags to control the script's behavior.

# If True, load a pre-trained model from `model_file`. If False, create a new model.
load_model_flag = False
# If True, train the model. If False, only load and evaluate it.
train_model_flag = True
# Number of epochs (passes through the entire training dataset) for training.
Nepochs = 400

# File paths
history_file = './NNModels/IAW_history_pytorch.npy'
model_file = "./NNModels/DeepIAW_pytorch.pth"
data_file = './data/Skw_features_532nm_MagPy_v1.1.csv'

# Ensure the NNModels directory exists
os.makedirs('./NNModels', exist_ok=True)

def appendHist(h1, h2):
    """
    Appends a new history dictionary to an existing one.

    This is useful for continuing the training of a model and keeping a complete 
    record of the loss and metrics over all training sessions.

    Args:
        h1 (dict): The existing history dictionary. Can be empty.
        h2 (dict): The new history dictionary to append.

    Returns:
        dict: The combined history dictionary.
    """
    if not h1:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest

def standardise_data(train_X, test_X, train_Y, test_Y):
    """
    Performs data standardization on training and testing sets.

    Standardization (or Z-score normalization) rescales the data to have a mean of 0 
    and a standard deviation of 1. This is important for many machine learning algorithms, 
    including neural networks, as it helps to stabilize training and improve convergence.

    Args:
        train_X, test_X, train_Y, test_Y: The training and testing data splits.

    Returns:
        tuple: A tuple containing the scaled data sets and the fitted output scaler, 
               which is needed to inverse-transform the predictions back to the original scale.
    """
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    # Fit the scalers only on the training data to avoid information leakage from the test set
    input_scaler.fit(train_X)
    output_scaler.fit(train_Y)

    # Apply the transformation to both training and testing data
    train_X = input_scaler.transform(train_X)
    train_Y = output_scaler.transform(train_Y)
    test_X = input_scaler.transform(test_X)
    test_Y = output_scaler.transform(test_Y)
    return train_X, test_X, train_Y, test_Y, output_scaler

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)

def get_model(load, input_size, output_size):
    """
    Loads a PyTorch model from a file or creates a new one.

    The new model is a simple Multi-Layer Perceptron (MLP).

    Args:
        load (bool): If True, load the model from `model_file`.
        input_size (int): The number of features in the input data.
        output_size (int): The number of features in the output data.

    Returns:
        tuple: A tuple containing the PyTorch model and its training history.
    """
    model = MLP(input_size, output_size)
    history = {}
    if load and os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        model.load_state_dict(torch.load(model_file))
        if os.path.exists(history_file):
            history = np.load(history_file, allow_pickle='TRUE').item()
    else:
        print("Creating a new model.")

    print(model)
    return model, history

# --- Main Script ---

# 1. Load and Prepare Data
print(f"Loading data from {data_file}")
data = np.loadtxt(data_file, delimiter=',')
# Input features are in columns 2 to 8
X = data[:, 2:9]
# Output targets are in columns 9 onwards
Y = data[:, 9:]

# 2. Split into Training and Testing Sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=42)

# 3. Standardize the Data
train_X, test_X, train_Y, test_Y, output_scaler = standardise_data(train_X, test_X, train_Y, test_Y)

# Convert data to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y = torch.tensor(test_Y, dtype=torch.float32)

input_size = train_X.shape[1]
output_size = train_Y.shape[1]

# 4. Get the Model
model, history = get_model(load_model_flag, input_size, output_size)

# 5. Train the Model
if train_model_flag:
    print(f"Training model for {Nepochs} epochs...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    train_history = {'loss': [], 'val_loss': []}

    for epoch in range(Nepochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_Y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(test_X)
            val_loss = criterion(val_outputs, test_Y)
        
        train_history['loss'].append(loss.item())
        train_history['val_loss'].append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{Nepochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Save the trained model and the combined history
    print(f"Saving model to {model_file}")
    torch.save(model.state_dict(), model_file)
    history = appendHist(history, train_history)
    np.save(history_file, history)
    print("Training complete and model saved.")

# 6. Evaluate and Plot Results

# Plot the training and validation loss from the history
fig = plt.figure(dpi=200, figsize=(6, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

if history and 'loss' in history:
    ax1.semilogx(np.arange(1, len(history['loss']) + 1), history['loss'], label='Train Loss')
    ax1.semilogx(np.arange(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
    ax1.legend(frameon=False)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    if len(history['loss']) > 0:
        ax1.set_xlim(1, len(history['loss']))
    ax1.set_title('Model Training History')

# Make predictions on the test set
model.eval()
with torch.no_grad():
    pred_y = model(test_X).numpy()

# Inverse-transform the standardized data and predictions to get them back to the original scale
test_Y = output_scaler.inverse_transform(test_Y.numpy())
pred_y = output_scaler.inverse_transform(pred_y)

def compare_plot(ax, pred, test, title):
    """Helper function to create a prediction vs. truth plot."""
    ax.plot(test, pred, 'bo', alpha=0.1, mew=0, ms=2.0)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')

# Create comparison plots for each of the output features
compare_plot(ax2, pred_y[:, 0], test_Y[:, 0], 'Output 1: Te')
compare_plot(ax3, pred_y[:, 1], test_Y[:, 1], 'Output 2: Ti')
compare_plot(ax4, pred_y[:, 2], test_Y[:, 2], 'Output 3: Zeff')

fig.tight_layout()
plt.show()
