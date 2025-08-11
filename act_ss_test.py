from tensorflow import keras, Session, ConfigProto
from tabulate import tabulate
import sklearn.metrics as sk
import numpy as np
import argparse
import os

# Add command prompt settings functionality for experiment parallelization
parser = argparse.ArgumentParser(description='Federated averaging experiment setup script.')
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-d', '--device', help='<Required> Set device on which experiments need to be run (-1 for CPU) (0-n for GPU 0-n)', required=True)
parser.add_argument('-t_f', '--test_files', type=int, nargs='+', help='<Required> Test files to be used in the experiment', required=True)
args = parser.parse_args()

# CUDA console output, visible devices settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# Limit GPU memory allocation
session_config = ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
keras.backend.set_session(Session(config=session_config))

# Model data
acc_test, gyr_test, label_test = [], [], []
for x in args.test_files:
	gyr_input, acc_input, labels = \
		np.load('datasets/MobiAct/gyr_seg_' + str(x) + '.npy').astype(np.float32), \
		np.load('datasets/MobiAct/acc_seg_' + str(x) + '.npy').astype(np.float32), \
		np.load('datasets/MobiAct/lb_' + str(x) + '.npy')

	acc_test.append(acc_input)
	gyr_test.append(gyr_input)
	label_test.append(labels)
acc_test, gyr_test, label_test = np.concatenate(acc_test), np.concatenate(gyr_test), np.concatenate(label_test)

ae_model = keras.models.load_model('models/' + args.model_name + '_selfsupervised.hdf5')
f = ae_model.layers[2].predict([acc_test, gyr_test])[0]

del ae_model
keras.backend.clear_session()

act_model = keras.models.load_model('models/' + args.model_name + '.hdf5')
y_pred = act_model.predict(f, batch_size=10)

precision, recall, f_score, _ = sk.precision_recall_fscore_support(np.argmax(label_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
acc = sk.accuracy_score(np.argmax(label_test, axis=1), np.argmax(y_pred, axis=1))
kappa_score = sk.cohen_kappa_score(np.argmax(label_test, axis=1), np.argmax(y_pred, axis=1))

print(str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + str(kappa_score))
# print(tabulate([[acc, precision, recall, f_score, kappa_score]], headers=['Accuracy', 'Precision', 'Recall', 'F-score', 'Kappa score']))
