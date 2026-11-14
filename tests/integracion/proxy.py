"""Proxy TCP congelable para simular un servidor de base de datos que deja de responder.

Reenvía el tráfico entre el cliente y PostgreSQL. ``congelar()`` deja de
reenviar (retiene los bytes) sin cerrar los sockets: para el cliente es igual
que un servidor colgado o una red cortada, el caso que ``statement_timeout`` no
puede resolver porque la consulta nunca llega o la respuesta nunca vuelve.

``tras(patron, modo)`` (etapa 7) actúa sobre UNA conexión: la primera cuyo cliente
envía ``patron``. ``modo="perder_respuesta"`` reenvía ese mensaje al servidor y retiene
desde entonces todas sus respuestas (un COMMIT aplicado cuya respuesta se pierde);
``modo="retener"`` retiene el mensaje sin enviarlo (el servidor nunca lo recibe). Las
conexiones nuevas no se ven afectadas (la verificación posterior funciona).
"""

from __future__ import annotations

import socket
import threading
import time


def _cerrado_por_el_otro(s: socket.socket) -> bool:
    try:
        return s.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True


class ProxyCongelable:
    def __init__(self, destino_host: str, destino_puerto: int):
        self.destino = (destino_host, destino_puerto)
        self._congelado = threading.Event()
        self._cerrado = threading.Event()
        self._sockets: list[socket.socket] = []
        self._tras: tuple[bytes, str] | None = None
        self._afectada: tuple | None = None
        self.disparado = threading.Event()
        self._servidor = socket.create_server(("127.0.0.1", 0))
        self.puerto = self._servidor.getsockname()[1]
        threading.Thread(target=self._aceptar, daemon=True).start()

    def congelar(self) -> None:
        self._congelado.set()

    def descongelar(self) -> None:
        self._congelado.clear()

    def tras(self, patron: bytes, modo: str) -> None:
        assert modo in ("perder_respuesta", "retener")
        self._tras = (patron, modo)
        self._afectada = None
        self.disparado.clear()

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
            par = (cliente, destino)
            threading.Thread(target=self._reenviar, args=(cliente, destino, par, True), daemon=True).start()
            threading.Thread(target=self._reenviar, args=(destino, cliente, par, False), daemon=True).start()

    def _retenida(self, par, de_cliente: bool) -> bool:
        if self._afectada is not par or self._tras is None:
            return False
        return (self._tras[1] == "retener") == de_cliente

    def _reenviar(self, origen: socket.socket, destino: socket.socket, par=None, de_cliente=False) -> None:
        try:
            while True:
                datos = origen.recv(65536)
                if not datos:
                    break
                if (de_cliente and self._tras is not None and self._afectada is None
                        and self._tras[0] in datos):
                    self._afectada = par
                    self.disparado.set()
                while ((self._congelado.is_set() or self._retenida(par, de_cliente))
                       and not self._cerrado.is_set()):
                    time.sleep(0.02)                      # retiene los bytes mientras está congelado
                    if self._retenida(par, de_cliente) and de_cliente and _cerrado_por_el_otro(origen):
                        break                             # el cliente cortó: el servidor revierte
                else:
                    destino.sendall(datos)
                    continue
                break
        except OSError:
            pass
        finally:
            for s in (origen, destino):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
