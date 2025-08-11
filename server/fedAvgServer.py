import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import json_tricks
import requests
import aiohttp
import asyncio
import random
import base64


class FedAvgServer:

	def __init__(self, fl_plan):
		self.plan = fl_plan
		self.model_path = 'models/' + self.plan[0].name + '.hdf5'
		self.last_visited_clients = []

	def __enter__(self):
		if asyncio.get_event_loop().is_closed():
			asyncio.set_event_loop(asyncio.new_event_loop())
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		asyncio.get_event_loop().close()

	# Train functions

	def simulate_train_model(self, split_client_data, debug=False):
		print("Setting up training instance...")
		train_loss_metrics = []

		# Save compiled model
		self.plan[0].save(filepath=self.model_path, overwrite=True, include_optimizer=True)

		# Execute communication rounds
		for t in range(0, self.plan[1]["communication_rounds"]):
			print("Communication round " + str(t) + " was started...")

			# Phase 1: client selection
			split_client_data_fraction = self.reservoir_sample(split_client_data, self.plan[1]["worker_nr"])

			# Phase 2: client training
			cumulative_train_response = [[np.zeros(weight_list.shape).astype(np.float32) for weight_list in self.plan[0].get_weights()], 0]
			metric_responses = []

			for client in split_client_data_fraction:
				keras.backend.clear_session()
				self.plan[0] = keras.models.load_model(self.model_path)
				weights = self.plan[0].get_weights()
				n = 0

				if len(self.plan[0].inputs) == 1:
					n = round((len(client[0]) / self.plan[1]["batch_size"]) * self.plan[1]["epochs"])
				else:
					for model_input in client[0]:
						n += round((len(model_input) / self.plan[1]["batch_size"]) * self.plan[1]["epochs"])

				history = self.plan[0].fit(client[0], client[1], batch_size=self.plan[1]["batch_size"], epochs=self.plan[1]["epochs"], verbose=0)
				return_weights = [np.subtract(x, y) * n for x, y in zip(self.plan[0].get_weights(), weights)]

				cumulative_train_response[0] = np.add(cumulative_train_response[0], return_weights)
				cumulative_train_response[1] += n
				metric_responses.append(history.history["loss"])

			# Phase 3: model update
			train_loss_metrics.extend(np.mean(metric_responses, axis=0).tolist())
			average_update = [np.divide(x, cumulative_train_response[1]) for x in cumulative_train_response[0]]

			keras.backend.clear_session()
			self.plan[0] = keras.models.load_model(self.model_path)
			old_weights = self.plan[0].get_weights()
			new_weights = [np.add(x, y) for x, y in zip(old_weights, average_update)]
			self.plan[0].set_weights(new_weights)
			self.plan[0].save(filepath=self.model_path, overwrite=True, include_optimizer=True)

		print("Training has finished and new model has been saved to disk at relative path: " + self.model_path)
		if debug is True:
			plt.figure()
			plt.plot([(i + 1) * self.plan[1]["batch_size"] for i in range(0, len(train_loss_metrics))], train_loss_metrics, label="Average train loss at federated clients")
			plt.title("Loss mitigation over time", fontsize=14)
			plt.xlabel("Input samples ran through network (nr of samples)", fontsize=12)
			plt.ylabel("Train loss (categorical cross-entropy)", fontsize=12)
			plt.legend()
			plt.show()

		return self.model_path, train_loss_metrics

	def train_model(self, debug=False):
		print("Setting up training instance...")
		train_loss_metrics = []

		# Save compiled model
		self.plan[0].save(filepath=self.model_path, overwrite=True, include_optimizer=True)

		# Execute communication rounds
		for t in range(0, self.plan[1]["communication_rounds"]):
			print("Communication round " + str(t) + " was started...")

			# Phase 1: selection from resource directory server
			visited_clients, unvisited_clients, required_workers = self.resource_directory_selection()
			if required_workers is False:
				print("Not enough clients were found to participate in the communication round. Going to the next round...")
				continue

			# Phase 2: client training configuration
			responses = self.async_client_configuration(visited_clients, unvisited_clients)

			# Phase 3: model update
			invalid_updates, total_updates = 0, len(responses)
			for response in responses:
				if isinstance(response[0], (asyncio.TimeoutError, asyncio.CancelledError)):
					invalid_updates += 1
					unvisited_clients[:] = [element for element in unvisited_clients if element != response[1]]
					responses.remove(response)

			if (invalid_updates / total_updates) > self.plan[1]["worker_dropout_tolerance"]:
				print("Unfortunately the client dropout rate was too high, moving to next communication round...")
				if unvisited_clients:
					self.last_visited_clients.extend(unvisited_clients)
				continue

			# Gather debug statistics
			train_loss_metrics.extend(np.mean([json_tricks.loads(response[0])[2] for response in responses], axis=0).tolist())

			# Computing weighted model average
			all_client_weights, all_client_update_weight = np.sum([json_tricks.loads(response[0])[0] for response in responses], axis=0), np.sum([json_tricks.loads(response[0])[1] for response in responses])
			average_update = [np.divide(x, all_client_update_weight) for x in all_client_weights]

			old_weights = self.plan[0].get_weights()
			new_weights = [np.add(x, y) for x, y in zip(old_weights, average_update)]
			self.plan[0].set_weights(new_weights)
			self.plan[0].save(filepath=self.model_path, overwrite=True, include_optimizer=True)

			if unvisited_clients:
				self.last_visited_clients.extend(unvisited_clients)

		print("Training has finished and new model has been saved to disk at relative path: " + self.model_path)
		if debug is True:
			plt.figure()
			plt.plot([(i + 1) * self.plan[1]["batch_size"] for i in range(0, len(train_loss_metrics))], train_loss_metrics, label="Average train loss at federated clients")
			plt.title("Loss mitigation over time", fontsize=14)
			plt.xlabel("Input samples ran through network (nr of samples)", fontsize=12)
			plt.ylabel("Train loss (categorical cross-entropy)", fontsize=12)
			plt.legend()
			plt.show()

		return self.model_path, train_loss_metrics

	# Simulate train utility functions

	# Adopted from the accepted answer found at https://stackoverflow.com/questions/2612648/reservoir-sampling
	def reservoir_sample(self, iterator, required_sample_nr):
		result = []
		n = 0

		for element in iterator:
			n = n + 1
			if len(result) < required_sample_nr:
				result.append(element)
			else:
				s = int(random.random() * n)
				if s < required_sample_nr:
					result[s] = element

		return result

	# Train utility functions

	def resource_directory_selection(self):
		directory_info = {
			"request": "client_request",
			"data": [self.plan[1]["data_function"]],
			"required_clients": self.plan[1]["worker_nr"]
		}
		req = requests.post(url="http://mscthesis.duckdns.org:8000", data=json_tricks.dumps(directory_info), headers={'content-type': 'application/json'})

		clients = json_tricks.loads(req.text)
		visited_clients = list(set(clients).intersection(self.last_visited_clients))
		unvisited_clients = list(set(clients) - set(self.last_visited_clients))

		required_workers = True
		if len(clients) < self.plan[1]["worker_nr"]:
			required_workers = False

		return [visited_clients, unvisited_clients, required_workers]

	def async_client_configuration(self, visited_clients, unvisited_clients):
		with open(self.model_path, 'rb') as f:
			model_string = base64.encodebytes(f.read()).decode('ascii')

		data_for_unvisited_client = json_tricks.dumps([model_string, self.plan[1]])
		self.plan[1]["model_name"] = self.plan[0].name
		data_for_visited_client = json_tricks.dumps([self.plan[0].get_weights(), self.plan[1]])
		del self.plan[1]["model_name"]

		# Phase 2: asynchronous client configuration
		dispatch_to_visited_clients = asyncio.gather(*[self.dispatch_to_client(client, data_for_visited_client) for client in visited_clients], return_exceptions=True)
		dispatch_to_unvisited_clients = asyncio.gather(*[self.dispatch_to_client(client, data_for_unvisited_client) for client in unvisited_clients], return_exceptions=True)
		dispatch_to_all_clients = asyncio.gather(dispatch_to_visited_clients, dispatch_to_unvisited_clients, return_exceptions=True)
		responses = asyncio.get_event_loop().run_until_complete(dispatch_to_all_clients)

		return [response for response_layer in responses for response in response_layer]

	async def dispatch_to_client(self, client, data):
		async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.plan[1]["worker_process_timeout"])) as session:
			async with session.post(url='http://' + client + ':' + str(8001), data=data) as response:
				try:
					return [await response.text(), client]
				except (asyncio.TimeoutError, asyncio.CancelledError) as e:
					return [e, client]
