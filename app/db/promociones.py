"""Consultas de las promociones (etapa 6, DECISIONES.md §15). PostgreSQL es la fuente
de verdad del modelo vigente (E6-9): la decisión con resolución ``confirmada`` y la
mayor ``secuencia``.

Las sentencias usan el estilo de parámetros de psycopg2 (``%(nombre)s``) para
ejecutarse igual en la conexión dedicada de la concesión (psycopg2) y en el motor
de la API (``Connection.exec_driver_sql``).
"""

from __future__ import annotations

from dataclasses import dataclass

SQL_VIGENTE = """
SELECT d.decision_id, r.secuencia, d.version, d.run_id, d.ruta_artefacto, d.sha256_artefacto, d.epoca
FROM resoluciones_promocion r
JOIN decisiones_promocion d ON d.decision_id = r.decision_id
WHERE r.resultado = 'confirmada' AND d.nombre_modelo = %(nombre)s
ORDER BY r.secuencia DESC
LIMIT 1
"""

SQL_DECISIONES_SIN_RESOLVER = """
SELECT d.decision_id, d.version, d.epoca
FROM decisiones_promocion d
LEFT JOIN resoluciones_promocion r ON r.decision_id = d.decision_id
WHERE r.decision_id IS NULL AND d.nombre_modelo = %(nombre)s AND d.epoca < %(epoca)s
ORDER BY d.decision_id
"""

SQL_RESOLUCION_DE = """
SELECT r.resultado, r.secuencia FROM resoluciones_promocion r WHERE r.decision_id = %(decision_id)s
"""


@dataclass(frozen=True)
class Vigente:
    decision_id: int
    secuencia: int
    version: str
    run_id: str
    ruta_artefacto: str
    sha256_artefacto: str
    epoca: int

    @property
    def clave(self) -> tuple[int, int]:
        """Identidad de la promoción: dos decisiones con la misma versión y SHA son distintas."""
        return (self.decision_id, self.secuencia)


def _vigente_de_fila(fila) -> Vigente | None:
    if fila is None:
        return None
    return Vigente(int(fila[0]), int(fila[1]), str(fila[2]), str(fila[3]), str(fila[4]), str(fila[5]), int(fila[6]))


def leer_vigente_cursor(cursor, nombre: str) -> Vigente | None:
    """Con un cursor psycopg2 (conexión dedicada de la concesión)."""
    cursor.execute(SQL_VIGENTE, {"nombre": nombre})
    return _vigente_de_fila(cursor.fetchone())


def leer_vigente(conn, nombre: str) -> Vigente | None:
    """Con una ``Connection`` de SQLAlchemy (API). ``None`` = consulta exitosa sin filas."""
    return _vigente_de_fila(conn.exec_driver_sql(SQL_VIGENTE, {"nombre": nombre}).fetchone())
