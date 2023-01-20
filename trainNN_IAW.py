import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense,InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras_visualizer import visualizer 

load  = True
train = False
Nepochs = 100
history_file = './NNModels/IAW_history.npy'
model_file = "./NNModels/DeepIAW.h5"
data_file = './data/Skw_features_532nm_MagPy_v1.1.csv'

def appendHist(h1, h2):
	""" Combined two 'history' dictionarys """
	if h1 == {}:
		return h2
	else:
		dest = {}
		for key, value in h1.items():
			dest[key] = value + h2[key]
	return dest

def standardise_data(train_X, test_X, train_Y, test_Y):
	""" Perform data standardisation, i.e. linear transform to mean 0 and std dev 1 """
	input_scaler  = StandardScaler()
	output_scaler = StandardScaler()

	input_scaler.fit(train_X)
	output_scaler.fit(train_Y)

	train_X = input_scaler.transform(train_X)
	train_Y = output_scaler.transform(train_Y)
	test_X = input_scaler.transform(test_X)
	test_Y = output_scaler.transform(test_Y)
	return train_X, test_X, train_Y, test_Y, output_scaler

def get_model(load):
	""" Load NN model from file or create it 
	    we will use a MSE loss function and the adam optimizer (which is an adaptive variant of gradient descent)"""
	if(load):
		model = load_model("./NNModels/DeepIAW.h5")
		history = np.load(history_file,allow_pickle='TRUE').item()
	else:
		# define the keras model
		model = Sequential()
		model.add(InputLayer(input_shape=(input_size,)))
		model.add(Dense(32, input_shape=(input_size,), activation='relu'))
		model.add(Dense(32, activation='tanh'))
		model.add(Dense(output_size))

		model.summary()

		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
		history = {}
	return model,history

# Load data
data = np.loadtxt(data_file,delimiter=',')
X = data[:,2:9]
Y = data[:,9:]

# Split into training and test data
train_test_split_state = 24
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=train_test_split_state)

# Standardise data
train_X, test_X, train_Y, test_Y, output_scaler = standardise_data(train_X, test_X, train_Y, test_Y)

input_size = train_X.shape[1]
output_size = train_Y.shape[1]

# Load or construct NN model
model,history = get_model(load)

# fit model
if(train):
	train_history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=Nepochs, verbose=0)

	model.save(model_file)
	history = appendHist(history,train_history.history)
	np.save(history_file,history)

# visualizer(model, format='png', view=True)

# plot loss during training
fig = plt.figure(dpi=200,figsize=(3,6))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.semilogx(np.arange(1,len(history['loss'])+1),history['loss'], label='train')
ax1.semilogx(np.arange(1,len(history['loss'])+1),history['val_loss'], label='test')
ax1.legend(frameon=False)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_xlim(1,len(history['loss']))

pred_y = model.predict(test_X)

def compare_plot(ax,pred,test):
	ax.plot(test,pred,'bo',alpha=0.1,mew=0,ms=2.0)
	lims = [
	    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]

	ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
	ax.set_xlabel("Truth")
	ax.set_ylabel("Prediction")

test_Y = output_scaler.inverse_transform(test_Y)
pred_y = output_scaler.inverse_transform(pred_y)

compare_plot(ax2,pred_y[:,0],test_Y[:,0])
compare_plot(ax3,pred_y[:,1],test_Y[:,1])
compare_plot(ax4,pred_y[:,2],test_Y[:,2])

ax2.set_xlim(-0.5,0.5)
ax2.set_ylim(-0.5,0.5)
ax3.set_xlim(-0.05,0.5)
ax3.set_ylim(-0.05,0.5)
ax4.set_xlim(-0.2,0.2)
ax4.set_ylim(-0.2,0.2)

fig.tight_layout()

plt.show()

