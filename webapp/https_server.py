#!/usr/bin/env python3
"""Simple HTTPS server for SF Security Camera frontend"""

import http.server
import ssl
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIRECTORY)

server_address = ('0.0.0.0', PORT)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('cert.pem', 'key.pem')

httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"HTTPS Server running on https://0.0.0.0:{PORT}")
print(f"Access from other devices: https://<this-computer-ip>:{PORT}")
print("Note: You'll need to accept the self-signed certificate warning in your browser")

httpd.serve_forever()
