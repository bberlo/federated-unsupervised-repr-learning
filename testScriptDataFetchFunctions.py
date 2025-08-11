import sklearn.model_selection as sk
from tensorflow import keras
import numpy as np

# Train fetch functions


def fashion_mnist():

	# Import and pre-process input data
	(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
	x_train, y_train = np.reshape(x_train, (-1, 28, 28, 1)).astype(np.float32) / 255, keras.utils.to_categorical(y_train, 10)
	_, x_eval, _, y_eval = sk.train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, stratify=y_train)

	return [x_eval, y_eval]

# Simulate train fetch functions


def simulate_fashion_mnist_iid(total_clients):
	(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
	x_train, y_train = np.reshape(x_train, (-1, 28, 28, 1)).astype(np.float32) / 255, keras.utils.to_categorical(y_train, 10)
	x_train, x_eval, y_train, y_eval = sk.train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=42, stratify=y_train)
	x_train, y_train = np.array_split(x_train, total_clients), np.array_split(y_train, total_clients)

	return [[[x, y] for x, y in zip(x_train, y_train)], [x_eval, y_eval]]


def simulate_cifar10_iid(total_clients):
	(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
	x_train, y_train = np.reshape(x_train, (-1, 32, 32, 3)).astype(np.float32) / 255, keras.utils.to_categorical(y_train, 10)
	x_train, x_eval, y_train, y_eval = sk.train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=42, stratify=y_train)
	x_train, y_train = np.array_split(x_train, total_clients), np.array_split(y_train, total_clients)

	return [[[x, y] for x, y in zip(x_train, y_train)], [x_eval, y_eval]]


# -------------------------------- Activity recognition data -------------------------------------------- #

def get_har_data(file_nr, total_clients, fed_learning_type, gyr_input_base_path, acc_input_base_path, label_base_path):
	fed_train_data, flat_acc_train_data, flat_gyr_train_data, train_labels, eval_data_acc, eval_data_gyr, eval_data_labels = \
		[], [], [], [], [], [], []

	for x in file_nr:
		gyr_input, acc_input, labels = \
			np.load(gyr_input_base_path + str(x) + '.npy').astype(np.float32), \
			np.load(acc_input_base_path + str(x) + '.npy').astype(np.float32), \
			np.load(label_base_path + str(x) + '.npy')

		acc_input_train, acc_input_eval, gyr_input_train, gyr_input_eval, labels_train, labels_eval = \
			sk.train_test_split(acc_input, gyr_input, labels, test_size=0.2, train_size=0.8, random_state=42, stratify=labels)

		acc_input_train, gyr_input_train, labels_train = \
			np.array_split(acc_input_train, round(total_clients / len(file_nr))), \
			np.array_split(gyr_input_train, round(total_clients / len(file_nr))), \
			np.array_split(labels_train, round(total_clients / len(file_nr)))

		for d, y, z in zip(acc_input_train, gyr_input_train, labels_train):
			if fed_learning_type == 'unsupervised':
				fed_train_data.append([[d, y], [d, y]])
			elif fed_learning_type == 'supervised':
				fed_train_data.append([[d, y], z])

			flat_acc_train_data.append(d)
			flat_gyr_train_data.append(y)
			train_labels.append(z)

		eval_data_acc.append(acc_input_eval)
		eval_data_gyr.append(gyr_input_eval)
		eval_data_labels.append(labels_eval)

	return fed_train_data, \
		np.concatenate(flat_acc_train_data), \
		np.concatenate(flat_gyr_train_data), \
		np.concatenate(train_labels), \
		np.concatenate(eval_data_acc), \
		np.concatenate(eval_data_gyr), \
		np.concatenate(eval_data_labels)


def get_har_ss_data(file_nr, total_clients, gyr_input_base_path, acc_input_base_path, gyr_label_base_path, acc_label_base_path):
	fed_train_data, eval_data_acc, eval_data_gyr, gyr_eval_data_labels, acc_eval_data_labels = \
		[], [], [], [], []

	for x in file_nr:
		gyr_input, acc_input, gyr_labels, acc_labels = \
			np.load(gyr_input_base_path + str(x) + '.npy').astype(np.float32), \
			np.load(acc_input_base_path + str(x) + '.npy').astype(np.float32), \
			np.load(gyr_label_base_path + str(x) + '.npy'), \
			np.load(acc_label_base_path + str(x) + '.npy')

		acc_input_train, acc_input_eval, gyr_input_train, gyr_input_eval, acc_labels_train, acc_labels_eval, gyr_labels_train, gyr_labels_eval = \
			sk.train_test_split(acc_input, gyr_input, acc_labels, gyr_labels, test_size=0.2, train_size=0.8, random_state=42, stratify=acc_labels)

		acc_input_train, gyr_input_train, acc_labels_train, gyr_labels_train = \
			np.array_split(acc_input_train, round(total_clients / len(file_nr))), \
			np.array_split(gyr_input_train, round(total_clients / len(file_nr))), \
			np.array_split(acc_labels_train, round(total_clients / len(file_nr))), \
			np.array_split(gyr_labels_train, round(total_clients / len(file_nr)))

		for d, y, z, k in zip(acc_input_train, gyr_input_train, acc_labels_train, gyr_labels_train):
			fed_train_data.append([[d, y], [z, k]])

		eval_data_acc.append(acc_input_eval)
		eval_data_gyr.append(gyr_input_eval)
		gyr_eval_data_labels.append(gyr_labels_eval)
		acc_eval_data_labels.append(acc_labels_eval)

	return fed_train_data, \
		np.concatenate(eval_data_acc), \
		np.concatenate(eval_data_gyr), \
		np.concatenate(acc_eval_data_labels), \
		np.concatenate(gyr_eval_data_labels)
