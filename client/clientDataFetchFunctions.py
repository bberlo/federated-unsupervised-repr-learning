import sklearn.model_selection as sk
from tensorflow import keras
import numpy as np
import random

# Note: the functions displayed below assume that data present at every client is only used for training.
# Validation and Testing are done at a central server.

# Only use this variable when splitting imported global data sets into equal chunks for every client in the system
total_clients_in_system = 10


def fashion_mnist_iid():

	# Import and pre-process input data
	(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
	x_train, y_train = np.reshape(x_train, (-1, 28, 28, 1)).astype(np.float32) / 255, keras.utils.to_categorical(y_train, 10)
	x_train, _, y_train, _ = sk.train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, stratify=y_train)

	# Split input data to simulate distributed data
	data_instance = random.randint(0, (total_clients_in_system - 1))
	x_train = np.array_split(x_train, total_clients_in_system)[data_instance]
	y_train = np.array_split(y_train, total_clients_in_system)[data_instance]

	return [x_train, y_train]


def fashion_mnist_non_iid():

	# Import, reshape and sort data on label
	(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
	x_train, _, y_train, _ = sk.train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, stratify=y_train)
	sort_indices = np.argsort(y_train)
	x_train, y_train = np.reshape(x_train, (-1, 28, 28, 1)).astype(np.float32) / 255, keras.utils.to_categorical(y_train, 10)
	x_train, y_train = x_train[sort_indices], y_train[sort_indices]

	# Split input into 2 data shards and concatenate shards to simulate distributed data
	data_instance_shards = random.sample(range(0, total_clients_in_system * 2), 2)
	x_train_shard1, y_train_shard1 = np.array_split(x_train, total_clients_in_system * 2)[data_instance_shards[0]], np.array_split(y_train, total_clients_in_system * 2)[data_instance_shards[0]]
	x_train_shard2, y_train_shard2 = np.array_split(x_train, total_clients_in_system * 2)[data_instance_shards[1]], np.array_split(y_train, total_clients_in_system * 2)[data_instance_shards[1]]
	x_train, y_train = np.concatenate((x_train_shard1, x_train_shard2), axis=0), np.concatenate((y_train_shard1, y_train_shard2), axis=0)

	return [x_train, y_train]
