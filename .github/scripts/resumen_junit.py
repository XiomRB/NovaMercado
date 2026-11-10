"""Resume un reporte JUnit de pytest en Markdown para el resumen del job de GitHub Actions."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(ruta: str, titulo: str) -> None:
    p = Path(ruta)
    if not p.exists():
        print(f"### {titulo}\n\nNo se generó el reporte (la ejecución falló antes de las pruebas).")
        return
    raiz = ET.parse(p).getroot()
    s = raiz if raiz.tag == "testsuite" else raiz.find("testsuite")
    total = int(s.get("tests", 0))
    fallos = int(s.get("failures", 0)) + int(s.get("errors", 0))
    omitidas = int(s.get("skipped", 0))
    estado = "✅ aprobado" if fallos == 0 else "❌ fallido"
    print(f"### {titulo}: {estado}\n")
    print("| Total | Aprobadas | Fallidas | Omitidas | Tiempo (s) |\n|---|---|---|---|---|")
    print(f"| {total} | {total - fallos - omitidas} | {fallos} | {omitidas} | {float(s.get('time', 0)):.1f} |")
    for caso in s.iter("testcase"):
        if caso.find("failure") is not None or caso.find("error") is not None:
            print(f"- ❌ `{caso.get('classname')}::{caso.get('name')}`")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
