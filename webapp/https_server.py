#!/usr/bin/env python3
"""Simple HTTPS server for SF Security Camera frontend"""

import http.server
import ssl
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIRECTORY)


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that suppresses SSL EOF errors (normal for video streaming)"""

    def handle(self):
        try:
            super().handle()
        except ssl.SSLEOFError:
            pass  # Client closed connection early - normal for video
        except BrokenPipeError:
            pass  # Client disconnected
        except ConnectionResetError:
            pass  # Client reset connection


server_address = ('0.0.0.0', PORT)
httpd = http.server.HTTPServer(server_address, QuietHTTPRequestHandler)

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('cert.pem', 'key.pem')

httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"HTTPS Server running on https://0.0.0.0:{PORT}")
print(f"Access from other devices: https://<this-computer-ip>:{PORT}")
print("Note: You'll need to accept the self-signed certificate warning in your browser")

httpd.serve_forever()
