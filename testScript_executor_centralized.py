from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import subprocess
import datetime

data_set_file_total = 61  # HHAR: 9, MobiAct: 61
kf = KFold(n_splits=5, shuffle=False, random_state=42)
act_data_c, act_ae_data_c, act_ss_data_c, act_ae_trans_data_c, act_ss_trans_data_c = [[0,0,0,0,0]], [], [], [], []

for (train_indices, test_indices), (hhar_tr_indices, hhar_te_indices) in zip(kf.split(list(range(0, data_set_file_total))), kf.split(list(range(0, 9)))):
	# -------- Fully supervised -----------
	"""
	subprocess.call([
		'python', 'act_centralized.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-m_n', 'act_centralized',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_test.py',
		'-d', '0',
		'-m_n', 'act_centralized',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_data_c.append(result)
	"""
	# -------- Unsupervised autoencoder -----------
	subprocess.call([
		'python', 'act_ae_centralized.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-m_n', 'act_ae_centralized',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_ae_test.py',
		'-d', '0',
		'-m_n', 'act_ae_centralized',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ae_data_c.append(result)

	# Note: feature transferability test. pre-train MobiAct, transfer to HHAR
	subprocess.call([
		'python', 'act_transfer_centralized.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-m_n', 'act_ae_centralized_transfer',
		'-p_m_n', 'act_ae_centralized_encoder',
		'-t_f'] + [str(i) for i in hhar_tr_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_transfer_test.py',
		'-d', '0',
		'-m_n', 'act_ae_centralized_transfer',
		'-t_f'] + [str(i) for i in hhar_te_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ae_trans_data_c.append(result)

	# -------- Self supervised -----------
	subprocess.call([
		'python', 'act_ss_centralized.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-m_n', 'act_ss_centralized',
		'-t_f'] + [str(i) for i in train_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_ss_test.py',
		'-d', '0',
		'-m_n', 'act_ss_centralized',
		'-t_f'] + [str(i) for i in test_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ss_data_c.append(result)

	# Note: feature transferability test. pre-train MobiAct, transfer to HHAR
	subprocess.call([
		'python', 'act_transfer_centralized.py',
		'-e_s', '2',
		'-b_s', '10',
		'-d', '0',
		'-r', '40',
		'-m_n', 'act_ss_centralized_transfer',
		'-p_m_n', 'act_ss_centralized_selfsupervised',
		'-t_f'] + [str(i) for i in hhar_tr_indices.tolist()])

	result = subprocess.check_output([
		'python', 'act_transfer_test.py',
		'-d', '0',
		'-m_n', 'act_ss_centralized_transfer',
		'-t_f'] + [str(i) for i in hhar_te_indices.tolist()])
	result = str(result, 'utf-8').strip().split(',')
	result = [float(i) for i in result]
	act_ss_trans_data_c.append(result)

act_data_c, act_ae_data_c, act_ss_data_c, act_ae_trans_data_c, act_ss_trans_data_c = \
	pd.DataFrame(data=np.array(act_data_c), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ae_data_c), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ss_data_c), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ae_trans_data_c), columns=['A', 'P', 'R', 'F', 'CK']), \
	pd.DataFrame(data=np.array(act_ss_trans_data_c), columns=['A', 'P', 'R', 'F', 'CK'])

# act_data_c.to_csv('figure_data/mobiact_act_centralized' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
# act_ae_data_c.to_csv('figure_data/mobiact_act_ae_centralized' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
# act_ss_data_c.to_csv('figure_data/mobiact_act_ss_centralized' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
act_ae_trans_data_c.to_csv('figure_data/mobiact_act_ae_centralized_trans' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
act_ss_trans_data_c.to_csv('figure_data/mobiact_act_ss_centralized_trans' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.csv')
