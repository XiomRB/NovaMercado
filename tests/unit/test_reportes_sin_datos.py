"""Los reportes publicados no contienen ventas reales ni secretos (etapa 9, E9-3).

Revisa los archivos actuales de ``reportes/``. Nunca lee ``raw/``.

* **Ventas en texto:** ``ventas[i].ventas=<n>``, "venta negativa <n>" y, en las entradas de
  calidad cuyo ``campo`` es ``ventas`` o ``ventas[i]...``, cualquier número del ``detalle``.
* **Ventas en la estructura JSON:** una clave ``ventas``, ``venta``, ``venta_real`` o ``real``
  con valor numérico (o texto con dígitos).
* **Exclusiones documentadas:** agregados de costo y métricas, ``venta_esperada``,
  ``cantidad_sugerida`` y ``en_promocion`` (salidas del modelo y promociones), índices, fechas e
  identificadores. En ``SINTETICOS`` (ejemplos hechos con ``tests/fixtures/ejemplos``) se
  admiten solo los valores de venta que aparecen en esas fixtures.
* **Secretos:** claves Fernet, tokens de GitHub, claves privadas y asignaciones de
  ``*PASSWORD``/``*CLAVE``/``*SECRET``/``*TOKEN`` con un valor literal que no sea un ejemplo de
  ``.env.example``.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
REPORTES = RAIZ / "reportes"
FIXTURES = RAIZ / "tests" / "fixtures" / "ejemplos"
SINTETICOS = ("reportes/entrega/api.json", "reportes/entrega/api.md")
EXTENSIONES = {".md", ".json", ".csv", ".txt"}

CLAVES_VENTA = {"ventas", "venta", "venta_real", "real"}
CAMPO_VENTAS = re.compile(r"^ventas(\[\d+\](\.\w+)?)?$")
TEXTO_VENTAS = (
    re.compile(r"ventas\[\d+\]\.ventas\s*=\s*['\"]?-?\d"),
    re.compile(r"venta negativa\s*['\"]?-?\d", re.IGNORECASE),
)
NUMERO = re.compile(r"-?\d+(?:[.,]\d+)*")

FERNET = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}=(?![A-Za-z0-9_=-])")
TOKENS = (
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
ASIGNACION = re.compile(r"\b([A-Z][A-Z0-9_]*(?:PASSWORD|CLAVE|SECRET|TOKEN)[A-Z0-9_]*)\s*[=:]\s*['\"]?([^\s'\"#,}]+)")


def ejemplos_env(raiz: Path = RAIZ) -> set[str]:
    """Valores de ejemplo de ``.env.example`` (no son credenciales)."""
    valores = set()
    for ln in (raiz / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            valores.add(ln.split("=", 1)[1].strip())
    return valores


def valores_fixtures(carpeta: Path = FIXTURES) -> set[str]:
    """Valores de venta (como texto) de las fixtures sintéticas de observaciones."""
    out: set[str] = set()

    def recorrer(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in CLAVES_VENTA and not isinstance(v, (dict, list)):
                    out.update(NUMERO.findall(str(v)))
                    out.add(str(v))
                recorrer(v)
        elif isinstance(o, list):
            for v in o:
                recorrer(v)

    for f in sorted(carpeta.rglob("*.json")):
        try:
            recorrer(json.loads(f.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return out


def _numeros(texto: str) -> list[str]:
    return NUMERO.findall(texto)


def hallazgos_texto(texto: str, permitidos: set[str] = frozenset()) -> list[str]:
    """Patrones de venta en texto libre (líneas de Markdown o cadenas de JSON)."""
    out = []
    for p in TEXTO_VENTAS:
        for m in p.finditer(texto):
            nums = _numeros(texto[m.start():m.end() + 12])
            if not nums or not set(nums) <= permitidos:
                out.append(f"texto:{p.pattern}")
    return out


def hallazgos_json(obj, permitidos: set[str] = frozenset()) -> list[str]:
    """Patrones de venta en la estructura: claves de venta numéricas y ``detalle`` de campos de ventas."""
    out: list[str] = []

    def recorrer(o, ruta):
        if isinstance(o, dict):
            campo = o.get("campo")
            if isinstance(campo, str) and CAMPO_VENTAS.match(campo):
                for clave in ("detalle", "detail"):
                    nums = set(_numeros(re.sub(r"\[\d+\]", "", str(o.get(clave, "")))))
                    if nums and not nums <= permitidos:
                        out.append(f"{ruta}.{clave}: número en un mensaje del campo {campo}")
            for k, v in o.items():
                if k in CLAVES_VENTA and not isinstance(v, (dict, list)) and v is not None \
                        and not isinstance(v, bool) and _numeros(str(v)) \
                        and not (str(v) in permitidos or set(_numeros(str(v))) <= permitidos):
                    out.append(f"{ruta}.{k}: valor numérico de venta")
                recorrer(v, f"{ruta}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                recorrer(v, f"{ruta}[{i}]")
        elif isinstance(o, str):
            out.extend(f"{ruta}: {h}" for h in hallazgos_texto(o, permitidos))

    recorrer(obj, "$")
    return out


def hallazgos_secretos(texto: str, ejemplos: set[str]) -> list[str]:
    out = []
    for m in FERNET.finditer(texto):
        try:
            if len(base64.urlsafe_b64decode(m.group(0))) == 32:
                out.append("clave Fernet")
        except (binascii.Error, ValueError):
            pass
    out += [f"token:{p.pattern[:20]}" for p in TOKENS if p.search(texto)]
    for m in ASIGNACION.finditer(texto):
        valor = m.group(2)
        if valor.startswith(("$", "<", "{")) or valor in ejemplos or valor.lower() in {"none", "null", "true", "false"}:
            continue
        out.append(f"asignación {m.group(1)}")
    return out


def revisar(ruta_rel: str, contenido: bytes, permitidos_sinteticos: set[str], ejemplos: set[str]) -> list[str]:
    """Todos los hallazgos de un archivo de ``reportes/`` (actual o histórico)."""
    try:
        texto = contenido.decode("utf-8")
    except UnicodeDecodeError:
        return []
    permitidos = permitidos_sinteticos if ruta_rel in SINTETICOS else set()
    out = hallazgos_secretos(texto, ejemplos)
    if ruta_rel.endswith(".json"):
        try:
            out += hallazgos_json(json.loads(texto), permitidos)
        except json.JSONDecodeError:
            out += hallazgos_texto(texto, permitidos)
    else:
        out += hallazgos_texto(texto, permitidos)
    return out


def archivos_reportes() -> list[Path]:
    return sorted(p for p in REPORTES.rglob("*") if p.is_file() and p.suffix in EXTENSIONES)


# ------------------------------------------------------------------ pruebas
def test_reportes_actuales_sin_ventas_ni_secretos():
    permitidos, ejemplos = valores_fixtures(), ejemplos_env()
    hallazgos = {}
    for p in archivos_reportes():
        rel = p.relative_to(RAIZ).as_posix()
        h = revisar(rel, p.read_bytes(), permitidos, ejemplos)
        if h:
            hallazgos[rel] = h[:5] + ([f"... {len(h) - 5} más"] if len(h) > 5 else [])
    assert not hallazgos, hallazgos


def test_detecta_los_patrones_conocidos():
    # mensajes como los del commit 6e60def (valores inventados para la prueba)
    obs = {"calidad": {"observaciones": [
        {"campo": "ventas[3].ventas", "accion": "reparar", "detalle": "'101,0' se interpretó como 101"},
        {"campo": "ventas[0].ventas", "accion": "advertir", "detalle": "venta negativa -7; no se guardó"}]}}
    assert len(hallazgos_json(obs)) >= 2
    det = {"detail": "la familia 'x' aparece repetida (ventas[1].ventas=5, ventas[2].ventas=6)"}
    assert hallazgos_json(det)
    assert hallazgos_json({"ventas": [{"familia_id": "x", "ventas": 12}]})
    assert hallazgos_json({"ventas": [{"familia_id": "x", "ventas": "1,305"}]})
    assert hallazgos_texto("| x | ventas[4].ventas=9 |")


def test_no_marca_lo_excluido():
    pred = {"calidad": {"observaciones": [
        {"campo": "familias[1].en_promocion", "detalle": "'2,0' se interpretó como 2"},
        {"campo": "tienda_id", "detalle": "'T06' se interpretó como 6"},
        {"campo": "ventas[3].ventas", "detalle": "'N,N' se interpretó como N"},
        {"campo": "ventas", "detalle": "no se recibieron ventas de: bebidas"}]},
        "predicciones": [{"venta_esperada": 857.01, "cantidad_sugerida": 1048}],
        "costos": {"servicio": 1.5, "referencia": 2.5}, "registros_guardados": 12}
    assert hallazgos_json(pred) == []


def test_sinteticos_solo_admiten_valores_de_las_fixtures():
    permitidos = valores_fixtures()
    assert "730" in permitidos                      # observacion_valida.json
    assert hallazgos_json({"ventas": 730}, permitidos) == []
    assert hallazgos_json({"ventas": 987654321}, permitidos)


def test_detecta_secretos():
    ejemplos = ejemplos_env()
    fernet = base64.urlsafe_b64encode(bytes(range(32))).decode()
    assert hallazgos_secretos(f"AIRFLOW_FERNET_KEY={fernet}", ejemplos)
    assert hallazgos_secretos("POSTGRES_PASSWORD=Xk9-real-valor", ejemplos)
    assert hallazgos_secretos("ghp_" + "a" * 36, ejemplos)
    assert hallazgos_secretos("POSTGRES_PASSWORD=cambie_esta_clave", ejemplos) == []
    assert hallazgos_secretos("POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?defina}", ejemplos) == []
    assert hallazgos_secretos("sha256 " + "a" * 64, ejemplos) == []
