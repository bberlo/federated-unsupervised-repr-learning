from tensorflow import keras
from math import inf
import tempfile
import shutil


class EarlyStoppingUtility:

	def __init__(self, patience, min_delta, mode, epochs_per_run):
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.epochs_per_run = epochs_per_run

		self.tempdir_path = tempfile.mkdtemp()
		self.current_patience_level = 0
		self.current_epoch = 0
		self.best_epoch = 0
		if self.mode == 'min':
			self.best_val_metric = inf
		elif self.mode == 'max':
			self.best_val_metric = 0

	def __del__(self):
		shutil.rmtree(self.tempdir_path)

	def update(self, val_metric, model):
		self.current_epoch += self.epochs_per_run

		if self.mode == 'min':
			if self.best_val_metric - val_metric <= self.min_delta:
				self.current_patience_level += 1
			else:
				self.current_patience_level = 0
				self.best_val_metric = val_metric
				self.best_epoch = self.current_epoch
				model.save(filepath=self.tempdir_path + model.name + '.hdf5', overwrite=True, include_optimizer=True)

		elif self.mode == 'max':
			if val_metric - self.best_val_metric <= self.min_delta:
				self.current_patience_level += 1
			else:
				self.current_patience_level = 0
				self.best_val_metric = val_metric
				self.best_epoch = self.current_epoch
				model.save(filepath=self.tempdir_path + model.name + '.hdf5', overwrite=True, include_optimizer=True)

		if self.current_patience_level > self.patience:
			return True
		else:
			return False

	def return_best_version(self, model):
		return self.tempdir_path + model.name + '.hdf5', self.best_epoch
