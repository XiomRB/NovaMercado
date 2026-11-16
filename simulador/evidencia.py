"""Evidencia inmutable de una ejecución (plan v3, punto 1).

* Cada archivo se escribe **una sola vez** (``escribir_unica``): temporal + ``os.link``
  (falla si el destino existe) y modo 0444.
* ``sellar`` copia ``respuestas.jsonl`` y genera ``evidencia.sha256`` **una sola vez**, ya
  con todos los archivos definitivos; después la carpeta queda en 0555. El manifiesto no
  existe antes del sellado ni se modifica después.
* ``verificar`` exige exactamente los archivos del manifiesto con su SHA-256.
* La limpieza de la ejecución nunca borra esta carpeta: solo ``borrar`` con confirmación.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

MANIFIESTO = "evidencia.sha256"
PREVIOS = ("costos.csv", "escenario.json", "origen.json", "ventas_referencia.csv")
SELLADOS = ("respuestas.jsonl",)


class EvidenciaInvalida(Exception):
    pass


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def escribir_unica(ruta: Path, datos: bytes) -> None:
    ruta = Path(ruta)
    if ruta.exists():
        raise EvidenciaInvalida(f"{ruta.name} ya existe: la evidencia se escribe una sola vez")
    tmp = ruta.with_name(f".{ruta.name}.tmp")
    with open(tmp, "xb") as fh:
        fh.write(datos)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(tmp, ruta)
    except FileExistsError as exc:
        raise EvidenciaInvalida(f"{ruta.name} ya existe: la evidencia se escribe una sola vez") from exc
    finally:
        tmp.unlink()
    os.chmod(ruta, 0o444)


def sellada(carpeta: Path) -> bool:
    return (Path(carpeta) / MANIFIESTO).exists()


def sellar(carpeta: Path, respuestas: Path) -> str:
    """Copia ``respuestas.jsonl`` y escribe el manifiesto (una sola vez). Devuelve el
    SHA-256 del manifiesto."""
    carpeta = Path(carpeta)
    if sellada(carpeta):
        raise EvidenciaInvalida("la evidencia ya está sellada")
    faltan = [a for a in PREVIOS if not (carpeta / a).is_file()]
    if faltan:
        raise EvidenciaInvalida(f"faltan archivos previos: {', '.join(faltan)}")
    escribir_unica(carpeta / "respuestas.jsonl", Path(respuestas).read_bytes())
    nombres = sorted(PREVIOS + SELLADOS)
    texto = "".join(f"{_sha((carpeta / n).read_bytes())}  {n}\n" for n in nombres)
    escribir_unica(carpeta / MANIFIESTO, texto.encode())
    os.chmod(carpeta, 0o555)
    return _sha(texto.encode())


def verificar(carpeta: Path) -> dict[str, str]:
    """Comprueba el manifiesto; devuelve {archivo: sha256}. Lanza ``EvidenciaInvalida``."""
    carpeta = Path(carpeta)
    if not sellada(carpeta):
        raise EvidenciaInvalida(f"{carpeta} no tiene {MANIFIESTO} (evidencia no sellada)")
    esperado: dict[str, str] = {}
    for linea in (carpeta / MANIFIESTO).read_text().splitlines():
        sha, _, nombre = linea.partition("  ")
        if len(sha) != 64 or not nombre or nombre in esperado:
            raise EvidenciaInvalida(f"línea inválida en el manifiesto: {linea!r}")
        esperado[nombre] = sha
    if set(esperado) != set(PREVIOS + SELLADOS):
        raise EvidenciaInvalida(f"el manifiesto no lista exactamente {sorted(PREVIOS + SELLADOS)}")
    presentes = {p.name for p in carpeta.iterdir()} - {MANIFIESTO}
    if presentes != set(esperado):
        raise EvidenciaInvalida(f"archivos distintos del manifiesto: {sorted(presentes ^ set(esperado))}")
    for nombre, sha in esperado.items():
        if _sha((carpeta / nombre).read_bytes()) != sha:
            raise EvidenciaInvalida(f"{nombre}: el SHA-256 no coincide con el manifiesto")
    return esperado


def huella_manifiesto(carpeta: Path) -> str:
    return _sha((Path(carpeta) / MANIFIESTO).read_bytes())


def borrar(carpeta: Path, confirmar: bool) -> None:
    if not confirmar:
        raise EvidenciaInvalida("borrar la evidencia exige --confirmar")
    carpeta = Path(carpeta)
    os.chmod(carpeta, stat.S_IRWXU)
    shutil.rmtree(carpeta)
