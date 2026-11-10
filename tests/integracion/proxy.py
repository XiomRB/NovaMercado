"""Proxy TCP congelable para simular un servidor de base de datos que deja de responder.

Reenvía el tráfico entre el cliente y PostgreSQL. ``congelar()`` deja de
reenviar (retiene los bytes) sin cerrar los sockets: para el cliente es igual
que un servidor colgado o una red cortada, el caso que ``statement_timeout`` no
puede resolver porque la consulta nunca llega o la respuesta nunca vuelve.
"""

from __future__ import annotations

import socket
import threading
import time


class ProxyCongelable:
    def __init__(self, destino_host: str, destino_puerto: int):
        self.destino = (destino_host, destino_puerto)
        self._congelado = threading.Event()
        self._cerrado = threading.Event()
        self._sockets: list[socket.socket] = []
        self._servidor = socket.create_server(("127.0.0.1", 0))
        self.puerto = self._servidor.getsockname()[1]
        threading.Thread(target=self._aceptar, daemon=True).start()

    def congelar(self) -> None:
        self._congelado.set()

    def descongelar(self) -> None:
        self._congelado.clear()

    def cortar_conexiones(self) -> None:
        """Cierra las conexiones abiertas (el servidor revierte sus transacciones)."""
        for s in list(self._sockets):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
        self._sockets.clear()

    def cerrar(self) -> None:
        self._cerrado.set()
        self.descongelar()
        self.cortar_conexiones()
        self._servidor.close()

    def _aceptar(self) -> None:
        while not self._cerrado.is_set():
            try:
                cliente, _ = self._servidor.accept()
            except OSError:
                return
            try:
                destino = socket.create_connection(self.destino, timeout=5)
                destino.settimeout(None)
            except OSError:
                cliente.close()
                continue
            self._sockets += [cliente, destino]
            threading.Thread(target=self._reenviar, args=(cliente, destino), daemon=True).start()
            threading.Thread(target=self._reenviar, args=(destino, cliente), daemon=True).start()

    def _reenviar(self, origen: socket.socket, destino: socket.socket) -> None:
        try:
            while True:
                datos = origen.recv(65536)
                if not datos:
                    break
                while self._congelado.is_set() and not self._cerrado.is_set():
                    time.sleep(0.02)                      # retiene los bytes mientras está congelado
                destino.sendall(datos)
        except OSError:
            pass
        finally:
            for s in (origen, destino):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
