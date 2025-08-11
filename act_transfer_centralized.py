from tensorflow import keras, Session, ConfigProto
import testScriptDataFetchFunctions
import sklearn.metrics as sk
import earlyStoppingUtility
import pandas as pd
import numpy as np
import datetime
import argparse
import os

# Add command prompt settings functionality for experiment parallelization
parser = argparse.ArgumentParser(description='Federated averaging experiment setup script.')
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-p_m_n', '--pretrained_model_name', help='<Required> Set pre-trained model name to be used in the experiment', required=True)
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-r', '--rounds', type=int, help='<Required> Amount of experimental runs (e.g. alternative for custom keras log implementation)', required=True)
parser.add_argument('-d', '--device', help='<Required> Set device on which experiments need to be run (-1 for CPU) (0-n for GPU 0-n)', required=True)
parser.add_argument('-t_f', '--train_files', type=int, nargs='+', help='<Required> Train files to be used in the experiment', required=True)
args = parser.parse_args()

# CUDA console output, visible devices settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# Limit GPU memory allocation
session_config = ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
keras.backend.set_session(Session(config=session_config))

# New test data directory creation
# TEST_DATA_FILE_PATH = 'figure_data/' + os.path.splitext(os.path.basename(__file__))[0] + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '/'
# if os.path.isdir(TEST_DATA_FILE_PATH) is False:
# 	os.mkdir(TEST_DATA_FILE_PATH)

# Model instructions
instructions = {
	"runs": 1,
	"batch_size": args.batch_size,
	"epochs": args.epoch_size
}

# Model data
_, acc_train, gyr_train, label_train, acc_eval, gyr_eval, label_eval = testScriptDataFetchFunctions.get_har_data(
	args.train_files,
	len(args.train_files),
	None,
	'datasets/HHAR_Data/gyr_seg_',
	'datasets/HHAR_Data/acc_seg_',
	'datasets/HHAR_Data/lb_')

# test_data_frame = pd.DataFrame(columns=['Run', 'Accuracy', 'Precision', 'Recall', 'F_score', 'Cohen_kappa_score'])

act_es_object = earlyStoppingUtility.EarlyStoppingUtility(5, 0, 'max', args.epoch_size)

# --------- SUPERVISED ACTIVITY CLASSIFIER -------------------------------------------------------------------
i_ac, i_gy = keras.layers.Input(shape=(400, 3)), keras.layers.Input(shape=(400, 3))
pretrained_model = keras.models.load_model('models/' + args.pretrained_model_name + '.hdf5')
encoded_o, _, _ = pretrained_model.layers[2]([i_ac, i_gy])
x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l2(0.0001))(encoded_o)
o = keras.layers.Dense(label_train.shape[1], activation='softmax')(x)
act_model = keras.models.Model([i_ac, i_gy], o, name=args.model_name)
act_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy')
# -----------------------------------------------------------------------------------------------------------------

# Centralized supervised classifier training
for run_checkpoint in range(0, args.rounds, instructions["runs"]):
	act_model.fit([acc_train, gyr_train], label_train, batch_size=instructions["batch_size"], epochs=instructions["epochs"], verbose=2)
	y_pred = act_model.predict([acc_eval, gyr_eval], batch_size=instructions["batch_size"])

	# precision, recall, f_score, _ = sk.precision_recall_fscore_support(np.argmax(label_eval, axis=1), np.argmax(y_pred, axis=1), average='weighted')
	acc = sk.accuracy_score(np.argmax(label_eval, axis=1), np.argmax(y_pred, axis=1))
	# kappa_score = sk.cohen_kappa_score(np.argmax(label_eval, axis=1), np.argmax(y_pred, axis=1))

	# test_data_frame = test_data_frame.append({
	# 	'Run': run_checkpoint,
	# 	'Accuracy': acc,
	# 	'Precision': precision,
	# 	'Recall': recall,
	# 	'F_score': f_score,
	# 	'Cohen_kappa_score': kappa_score
	# }, ignore_index=True)

	patience_overflow = act_es_object.update(acc, act_model)
	if patience_overflow:
		break

path, epochs = act_es_object.return_best_version(act_model)
del act_model
keras.backend.clear_session()

best_act_model = keras.models.load_model(path)
best_act_model.save(filepath='models/' + best_act_model.name + '.hdf5', overwrite=True, include_optimizer=True)
# print('Best model version reached at: ' + str(epochs) + ' epochs')

# test_data_frame.to_csv(TEST_DATA_FILE_PATH + "B=" + str(instructions["batch_size"]) + "_E=" + str(instructions["epochs"]) + ".csv", index=False)
