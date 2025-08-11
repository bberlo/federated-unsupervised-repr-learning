from http.server import HTTPServer, BaseHTTPRequestHandler
import clientDataFetchFunctions
from tensorflow import keras
import numpy as np
import json_tricks
import requests
import atexit
import socket
import base64
import h5py
import io


def resource_directory_access(request, headers, types):

	with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
		s.connect(("8.8.8.8", 80))
		ip = s.getsockname()[0]

	directory_info = {
		"request": request,
		"local_ip": ip
	}
	if request == "client_add":
		directory_info["data"] = types
		directory_info["is_busy"] = False

	requests.post(url="http://mscthesis.duckdns.org:8000", data=json_tricks.dumps(directory_info), headers=headers)


class ClientHandler(BaseHTTPRequestHandler):

	f_path_start = "models/"

	def do_POST(self):
		content_length = int(self.headers["Content-Length"])
		body = self.rfile.read(content_length).decode("utf-8")
		self.send_response(200)
		self.end_headers()

		model_object, model_instructions = json_tricks.loads(body)
		self.wfile.write(bytes(json_tricks.dumps(self.train_model(model_object, model_instructions)), "utf-8"))

	def train_model(self, retrieved_object, instructions):
		keras.backend.clear_session()

		if "model_name" in instructions:
			model_path = self.f_path_start + instructions["model_name"] + ".hdf5"
			model = keras.models.load_model(model_path)
			model.set_weights(retrieved_object)
			weights = retrieved_object
		else:
			model_bytes = base64.decodebytes(retrieved_object.encode("ascii"))
			feed_buffer = io.BytesIO(model_bytes)
			model_file_instance = h5py.File(feed_buffer)
			model = keras.models.load_model(model_file_instance)
			weights = model.get_weights()

		model_feed_dict, n = getattr(clientDataFetchFunctions, instructions["data_function"])(), 0

		if len(model.inputs) == 1:
			n = round((len(model_feed_dict[0]) / instructions["batch_size"]) * instructions["epochs"])
		else:
			for model_input in model_feed_dict[0]:
				n += round((len(model_input) / instructions["batch_size"]) * instructions["epochs"])

		resource_directory_access("client_alter_busy_status", {"content-type": "application/json"}, None)
		history = model.fit(model_feed_dict[0], model_feed_dict[1], batch_size=instructions["batch_size"], epochs=instructions["epochs"], verbose=2)
		resource_directory_access("client_alter_busy_status", {"content-type": "application/json"}, None)

		return_weights = [np.subtract(x, y) * n for x, y in zip(model.get_weights(), weights)]
		model_path = self.f_path_start + model.name + ".hdf5"
		model.save(filepath=model_path, overwrite=True, include_optimizer=True)

		return [return_weights, n, history.history["loss"]]


# Register service at resource directory, instantiate service de-registration at system exit
resource_directory_access("client_add", {"content-type": "application/json"}, [k for k, v in clientDataFetchFunctions.__dict__.items() if callable(v)])
atexit.register(resource_directory_access, request="client_delete", headers={"content-type": "application/json"}, types=None)

# Activate client access point for global model server
with HTTPServer(('', 8001), ClientHandler) as client_server:
	client_server.serve_forever()
