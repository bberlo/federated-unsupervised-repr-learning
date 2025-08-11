from tensorflow import keras, Session, ConfigProto
from architectures import backbone_ae
import testScriptDataFetchFunctions
import sklearn.metrics as sk
import earlyStoppingUtility
from math import floor
import pandas as pd
import numpy as np
import datetime
import argparse
import os

# Add command prompt settings functionality for experiment parallelization
parser = argparse.ArgumentParser(description='Federated averaging experiment setup script.')
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-r', '--rounds', type=int, help='<Required> Amount of experimental runs (e.g. alternative for custom keras log implementation)', required=True)
parser.add_argument('-d', '--device', help='<Required> Set device on which experiments need to be run (-1 for CPU) (0-n for GPU 0-n)', required=True)
parser.add_argument('-t_f', '--train_files', type=int, nargs='+', help='<Required> Train files to be used in the experiment', required=True)
args = parser.parse_args()

# CUDA console output, visible device settings
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
	'datasets/MobiAct/gyr_seg_',
	'datasets/MobiAct/acc_seg_',
	'datasets/MobiAct/lb_')

# test_data_frame = pd.DataFrame(columns=['Run', 'Accuracy', 'Precision', 'Recall', 'F_score', 'Cohen_kappa_score'])

ae_es_object = earlyStoppingUtility.EarlyStoppingUtility(5, 0, 'min', args.epoch_size)

# ---------- UNSUPERVISED AUTO-ENCODER MODEL ----------------------------------------------------------------------
backbone_net = backbone_ae.BackboneAE().get_model()
i_ac, i_gy = keras.layers.Input(shape=(400, 3)), keras.layers.Input(shape=(400, 3))
_, decoded_ac, decoded_gy = backbone_net([i_ac, i_gy])
ae_model = keras.models.Model([i_ac, i_gy], [decoded_ac, decoded_gy], name=args.model_name + '_encoder')
ae_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mean_squared_error')
# -----------------------------------------------------------------------------------------------------------------

# Centralized unsupervised auto-encoder training
for _ in range(0, args.rounds, instructions["runs"]):
	ae_model.fit([acc_train, gyr_train], [acc_train, gyr_train], batch_size=instructions["batch_size"], epochs=instructions["epochs"], verbose=2)

	# Custom early stopping procedure
	decoded_pred = ae_model.predict([acc_eval, gyr_eval], batch_size=instructions["batch_size"])
	current_mse = floor(sum([np.mean(np.square(np.subtract(x, y))) for x, y in zip(decoded_pred, [acc_eval, gyr_eval])]) * 10000) / 10000.0

	patience_overflow = ae_es_object.update(current_mse, ae_model)
	if patience_overflow:
		break

path, epochs = ae_es_object.return_best_version(ae_model)
# print('Best model version reached at: ' + str(epochs) + ' epochs')

del ae_model
keras.backend.clear_session()

ae_model = keras.models.load_model(path)
ae_model.save(filepath='models/' + ae_model.name + '.hdf5', overwrite=True, include_optimizer=True)
backbone_bottleneck_f = ae_model.layers[2].predict([acc_train, gyr_train])[0]
backbone_bottleneck_val_f = ae_model.layers[2].predict([acc_eval, gyr_eval])[0]

del ae_model
keras.backend.clear_session()
del ae_es_object

act_es_object = earlyStoppingUtility.EarlyStoppingUtility(0, 0, 'max', args.epoch_size)

# --------- SUPERVISED ACTIVITY CLASSIFIER -------------------------------------------------------------------
backbone_bottleneck_input = keras.layers.Input(shape=(128, ))
x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001), activity_regularizer=keras.regularizers.l2(0.0001))(backbone_bottleneck_input)
o = keras.layers.Dense(label_train.shape[1], activation='softmax')(x)
act_model = keras.models.Model(backbone_bottleneck_input, o, name=args.model_name)
act_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy')
# -----------------------------------------------------------------------------------------------------------------

# Centralized supervised classifier training
for run_checkpoint in range(0, args.rounds, instructions["runs"]):
	act_model.fit(backbone_bottleneck_f, label_train, batch_size=instructions["batch_size"], epochs=instructions["epochs"], verbose=2)
	y_pred = act_model.predict(backbone_bottleneck_val_f, batch_size=instructions["batch_size"])

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
