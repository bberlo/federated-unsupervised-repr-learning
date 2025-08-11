from http.server import HTTPServer, BaseHTTPRequestHandler
import json_tricks
import requests
import random
import socket


class ResourceDirectoryHandler(BaseHTTPRequestHandler):

	directory_list = []

	def do_POST(self):
		content_length = int(self.headers['Content-Length'])
		body = self.rfile.read(content_length).decode('utf-8')
		self.send_response(200)
		self.end_headers()
		result_dictionary = json_tricks.loads(body)
		self.request_resolver(result_dictionary)(result_dictionary)

	def client_add(self, result_dictionary):
		del result_dictionary["request"]
		self.directory_list.append(result_dictionary)

	def client_delete(self, result_dictionary):
		self.directory_list[:] = [element for element in self.directory_list if element["local_ip"] != result_dictionary["local_ip"]]

	def client_request(self, result_dictionary):
		request_list = [item["local_ip"] for item in self.directory_list if set(result_dictionary["data"]).issubset(item["data"]) and item["is_busy"] is False]
		self.wfile.write(bytes(json_tricks.dumps(self.reservoir_sample(request_list, result_dictionary["required_clients"])), "utf-8"))

	def client_alter_busy_status(self, result_dictionary):
		for element in self.directory_list:
			if element["local_ip"] == result_dictionary["local_ip"]:
				element["is_busy"] = not element["is_busy"]

	def request_resolver(self, result_dictionary):
		return {
			"client_add": self.client_add,
			"client_delete": self.client_delete,
			"client_request": self.client_request,
			"client_alter_busy_status": self.client_alter_busy_status
		}[result_dictionary["request"]]

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


# Resolve ip address
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]

# DDNS update
requests.get('https://www.duckdns.org/update?domains=mscthesis&token=7e3bd465-90b4-4af4-8aa8-4b13a12e575c&ip=' + ip)

# Resource directory service setup
with HTTPServer(('', 8000), ResourceDirectoryHandler) as directory_server:
	directory_server.serve_forever()
