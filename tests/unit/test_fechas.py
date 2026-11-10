from datetime import date

import pytest

from app.calidad.fechas import FechaInvalida, interpretar_fecha

D = date(2025, 11, 17)


@pytest.mark.parametrize("entrada, esperado, accion", [
    ("2025-11-17", D, None),
    (" 2025-11-17 ", D, "reparar"),
    ("2025-11-17T00:00:00", D, "reparar"),               # E04
    ("2025-11-17T08:30:00Z", D, "reparar"),
    ("2025-11-17 08:30", D, "reparar"),
    ("2025-11-17T00:00:00.000-06:00", D, "reparar"),
    ("2025/11/17", D, "reparar"),
    ("2025-1-5", date(2025, 1, 5), "reparar"),
    ("20251117", D, "reparar"),
    ("17/11/2025", D, "reparar"),                        # E04: día > 12, única lectura
    ("17-11-2025", D, "reparar"),
    ("17.11.2025", D, "reparar"),
    ("11/17/2025", D, "reparar"),                        # solo puede ser MM/DD
    ("05/05/2025", date(2025, 5, 5), "reparar"),         # día == mes
    ("05/11/2025", date(2025, 11, 5), "advertir"),       # ambiguo -> DD/MM
    ("17/11/25", D, "advertir"),                         # año de dos dígitos
    ("17 de noviembre de 2025", D, "reparar"),
    ("17-nov-2025", D, "reparar"),
    ("17 Nov 2025", D, "reparar"),
    ("1 de setiembre de 2026", date(2026, 9, 1), "reparar"),
])
def test_interpreta_fechas(entrada, esperado, accion):
    fecha, acc, detalle = interpretar_fecha(entrada)
    assert fecha == esperado
    assert acc == accion
    assert (detalle is None) == (accion is None)


@pytest.mark.parametrize("entrada", [
    None, "", "   ", 20251117, 17.11, True, "mañana", "2025-02-30", "31/02/2025",
    "13/13/2025", "2025-13-01", "17 de brumario de 2025", "2025-11", "17/11", ["2025-11-17"],
])
def test_rechaza_fechas_invalidas(entrada):
    with pytest.raises(FechaInvalida):
        interpretar_fecha(entrada)


def test_no_depende_del_reloj():
    """Fechas muy lejanas en el futuro o el pasado se interpretan igual: el reloj no se consulta."""
    assert interpretar_fecha("2099-01-01")[0] == date(2099, 1, 1)
    assert interpretar_fecha("1999-01-01")[0] == date(1999, 1, 1)
