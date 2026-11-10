import math

import pytest

from app.calidad.numeros import NumeroInvalido, interpretar_numero, redondear


@pytest.mark.parametrize("entrada, esperado, con_formato", [
    (3, 3, False),
    (2.0, 2.0, False),
    ("3", 3, True),
    (" 3 ", 3, True),
    ("2.5", 2.5, True),
    ("1,234", 1234, True),          # E10: coma de miles
    ("1,292", 1292, True),          # E10 publicado
    ("12,345,678", 12345678, True),
    ("1,234.5", 1234.5, True),
    ("812,0", 812.0, True),          # E10: coma decimal
    ("812,50", 812.5, True),
    ("1,2345", 1.2345, True),        # 4 dígitos: no puede ser miles
    ("1.234.567", 1234567, True),    # varios puntos: solo miles
    ("1.234,5", 1234.5, True),       # formato europeo
    ("1 234", 1234, True),
    ("-4", -4, True),
    ("1.234", 1.234, True),          # convención local: punto decimal
])
def test_interpreta_formatos(entrada, esperado, con_formato):
    n = interpretar_numero(entrada)
    assert math.isclose(n.valor, esperado)
    assert (n.formato is not None) == con_formato


@pytest.mark.parametrize("entrada", [
    None, True, False, "", "abc", "tres", "1,23,4", "1..2", "12a", float("nan"),
    float("inf"), [], {}, "1,234,56", "--3",
])
def test_rechaza_no_numericos(entrada):
    with pytest.raises(NumeroInvalido):
        interpretar_numero(entrada)


@pytest.mark.parametrize("x, esperado", [(2.5, 3), (3.5, 4), (2.49, 2), (0.5, 1), (0, 0)])
def test_redondeo_aritmetico(x, esperado):
    assert redondear(x) == esperado
