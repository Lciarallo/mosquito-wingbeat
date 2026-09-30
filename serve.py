#!/usr/bin/env python3
"""Servidor HTTP local para visualização do estudo de classificação acústica de mosquitos."""

import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

class EstudoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_HEAD(self):
        if self.path in ('/', ''):
            self.send_response(302)
            self.send_header('Location', '/mosquito_wingbeat_estudo.html')
            self.end_headers()
            return
        super().do_HEAD()

    def do_GET(self):
        # Redireciona a raiz diretamente para o relatório HTML do estudo
        if self.path in ('/', ''):
            self.send_response(302)
            self.send_header('Location', '/mosquito_wingbeat_estudo.html')
            self.end_headers()
            return
        super().do_GET()

if __name__ == '__main__':
    # Permite reutilizar endereço de porta imediatamente caso reiniciado
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', PORT), EstudoHTTPRequestHandler) as httpd:
        print(f"Servidor HTTP iniciado em http://localhost:{PORT}/", flush=True)
        print(f"Estudo disponível em: http://localhost:{PORT}/mosquito_wingbeat_estudo.html", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor encerrado.", flush=True)
