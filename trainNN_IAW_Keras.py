"""
Demonstration of Training a Neural Network for Physics Data Analysis using TensorFlow and Keras.

This script provides a complete workflow for training and evaluating a Multi-Layer 
Perceptron (MLP) neural network using TensorFlow and Keras. The network is designed 
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
4.  **Model Architecture**: Defining a sequential Keras model with several dense (fully-connected) 
    layers and different activation functions (ReLU, tanh).
5.  **Training**: If training is enabled, the script compiles the model with a loss function 
    (Mean Squared Error) and an optimizer (Adam), then fits the model to the training data. 
    The model and its training history are saved.
6.  **Evaluation**: The script plots the training and validation loss over epochs to monitor 
    for overfitting. It then uses the trained model to make predictions on the unseen test 
    data and generates comparison plots to evaluate the model's accuracy.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Set these flags to control the script's behavior.

# If True, load a pre-trained model from `model_file`. If False, create a new model.
load_model_flag = True
# If True, train the model. If False, only load and evaluate it.
train_model_flag = False
# Number of epochs (passes through the entire training dataset) for training.
Nepochs = 100

# File paths
history_file = './NNModels/IAW_history.npy'
model_file = "./NNModels/DeepIAW.h5"
data_file = './data/Skw_features_532nm_MagPy_v1.1.csv'

def appendHist(h1, h2):
    """
    Appends a new Keras history dictionary to an existing one.

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

def get_model(load, input_size, output_size):
    """
    Loads a Keras model from a file or creates a new one.

    The new model is a simple Multi-Layer Perceptron (MLP) with two hidden layers.

    Args:
        load (bool): If True, load the model from `model_file`.
        input_size (int): The number of features in the input data.
        output_size (int): The number of features in the output data.

    Returns:
        tuple: A tuple containing the Keras model and its training history.
    """
    if load:
        print(f"Loading model from {model_file}")
        model = load_model(model_file)
        history = np.load(history_file, allow_pickle='TRUE').item()
    else:
        print("Creating a new model.")
        # Define the Keras Sequential model
        model = Sequential()
        # Input layer: specifies the shape of the input data.
        model.add(InputLayer(input_shape=(input_size,)))
        # Hidden layer 1: A fully-connected layer with 32 neurons and ReLU activation.
        model.add(Dense(32, activation='relu'))
        # Hidden layer 2: A fully-connected layer with 32 neurons and tanh activation.
        model.add(Dense(32, activation='tanh'))
        # Output layer: A linear layer with a neuron for each output feature.
        model.add(Dense(output_size))

        # Print a summary of the model architecture
        model.summary()

        # Compile the model, specifying the loss function and optimizer.
        # 'adam' is an efficient gradient descent variant.
        # 'mean_squared_error' is a common loss function for regression problems.
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        history = {}

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
# This is crucial to evaluate how well the model generalizes to new, unseen data.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=42)

# 3. Standardize the Data
train_X, test_X, train_Y, test_Y, output_scaler = standardise_data(train_X, test_X, train_Y, test_Y)

input_size = train_X.shape[1]
output_size = train_Y.shape[1]

# 4. Get the Model
model, history = get_model(load_model_flag, input_size, output_size)

# 5. Train the Model
if train_model_flag:
    print(f"Training model for {Nepochs} epochs...")
    train_history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=Nepochs, verbose=1)

    # Save the trained model and the combined history
    print(f"Saving model to {model_file}")
    model.save(model_file)
    history = appendHist(history, train_history.history)
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
    ax1.set_xlim(1, len(history['loss']))
    ax1.set_title('Model Training History')

# Make predictions on the test set
pred_y = model.predict(test_X)

# Inverse-transform the standardized data and predictions to get them back to the original scale
test_Y = output_scaler.inverse_transform(test_Y)
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
