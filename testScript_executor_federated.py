from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import subprocess
import datetime

data_set_file_total = 61  # HHAR: 9, MobiAct: 61
kf = KFold(n_splits=5, shuffle=False, random_state=42)
act_data_f, act_ae_data_f, act_ss_data_f, act_ae_trans_data_f, act_ss_trans_data_f = [[0,0,0,0,0]], [], [], [], []

for (train_indices, test_indices), (hhar_tr_indices, hhar_te_indices) in zip(kf.split(list(range(0, data_set_file_total))), kf.split(list(range(0, 9)))):
	# -------- Fully supervised -----------
	"""
	subprocess.call([
		'python', 'act_federated.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-f', '1',
		'-c', str(len(train_indices.tolist())),
		'-m_n', 'act_federated',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_test.py',
		'-d', '0',
		'-m_n', 'act_federated',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_data_f.append(result)
	"""
	# -------- Unsupervised autoencoder -----------
	subprocess.call([
		'python', 'act_ae_federated.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-f', '1',
		'-c', str(len(train_indices.tolist())),
		'-m_n', 'act_ae_federated',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_ae_test.py',
		'-d', '0',
		'-m_n', 'act_ae_federated',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ae_data_f.append(result)

	# Note: feature transferability test. pre-train MobiAct, transfer to HHAR
	subprocess.call([
		'python', 'act_transfer_federated.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-f', '1',
		'-c', str(len(hhar_tr_indices.tolist())),
		'-m_n', 'act_ae_federated_transfer',
		'-p_m_n', 'act_ae_federated_encoder',
		'-t_f'] + [str(i) for i in hhar_tr_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_transfer_test.py',
		'-d', '0',
		'-m_n', 'act_ae_federated_transfer',
		'-t_f'] + [str(i) for i in hhar_te_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ae_trans_data_f.append(result)

	# -------- Self supervised -----------
	subprocess.call([
		'python', 'act_ss_federated.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-f', '1',
		'-c', str(len(train_indices.tolist())),
		'-m_n', 'act_ss_federated',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_ss_test.py',
		'-d', '0',
		'-m_n', 'act_ss_federated',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ss_data_f.append(result)

	# Note: feature transferability test. pre-train MobiAct, transfer to HHAR
	subprocess.call([
		'python', 'act_transfer_federated.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-f', '1',
		'-c', str(len(hhar_tr_indices.tolist())),
		'-m_n', 'act_ss_federated_transfer',
		'-p_m_n', 'act_ss_federated_selfsupervised',
		'-t_f'] + [str(i) for i in hhar_tr_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_transfer_test.py',
		'-d', '0',
		'-m_n', 'act_ss_federated_transfer',
		'-t_f'] + [str(i) for i in hhar_te_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ss_trans_data_f.append(result)

act_data_f, act_ae_data_f, act_ss_data_f, act_ae_trans_data_f, act_ss_trans_data_f = \
	pd.DataFrame(data=np.array(act_data_f), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ae_data_f), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ss_data_f), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ae_trans_data_f), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ss_trans_data_f), columns=['A', 'P', 'R', 'F', 'CK'])

# act_data_f.to_csv('figure_data/mobiact_act_federated' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
# act_ae_data_f.to_csv('figure_data/mobiact_act_ae_federated' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
# act_ss_data_f.to_csv('figure_data/mobiact_act_ss_federated' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
act_ae_trans_data_f.to_csv('figure_data/mobiact_act_ae_federated_trans' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
act_ss_trans_data_f.to_csv('figure_data/mobiact_act_ss_federated_trans' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
