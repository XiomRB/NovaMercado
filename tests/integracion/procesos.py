"""Procesos auxiliares de las pruebas de concurrencia (se ejecutan con ``python -m``).

* ``tomar <json>``: toma la concesión, imprime ``{"epoca", "pid_bd"}`` y la mantiene
  sondeando cada ``sondeo`` s hasta que la pierde (sale con 6) o lo matan.
* ``intentar <json>``: intenta tomarla con ``espera``; imprime la época u ``ocupado``.
* ``supervisor <json>``: ``ejecutar_entrenamiento(**args)`` con los códigos de salida de
  ``entrenar_mlflow``.
"""

import json
import sys
import time


def main(argv: list[str]) -> int:
    accion, args = argv[0], json.loads(argv[1])
    from app.modelo.bloqueo import BloqueoOcupado, ConcesionEntrenamiento, ConcesionPerdida
    if accion == "tomar":
        c = ConcesionEntrenamiento(args["url"], modo="prueba", espera=args.get("espera", 0),
                                   vencimiento=args.get("vencimiento", 60))
        c.adquirir()
        print(json.dumps({"epoca": c.epoca, "pid_bd": c._pid}), flush=True)
        while True:
            time.sleep(args.get("sondeo", 1))
            if not c.sondear():
                print("perdida", flush=True)
                return 6
    if accion == "intentar":
        try:
            with ConcesionEntrenamiento(args["url"], modo="prueba", espera=args.get("espera", 0),
                                        vencimiento=args.get("vencimiento", 60), reintento=0.5) as c:
                print(json.dumps({"epoca": c.epoca}), flush=True)
                time.sleep(args.get("mantener", 0))
            return 0
        except BloqueoOcupado as exc:
            print(json.dumps({"ocupado": str(exc)}), flush=True)
            return 5
    if accion == "supervisor":
        import logging
        logging.basicConfig(level=logging.INFO)
        from app.modelo.ciclo_entrenamiento import ejecutar_entrenamiento
        from app.modelo.promocion import PromocionIndeterminada
        try:
            r = ejecutar_entrenamiento(**args)
        except BloqueoOcupado:
            return 5
        except ConcesionPerdida:
            return 6
        except PromocionIndeterminada:
            return 4
        print("RESULTADO " + r.como_json(), flush=True)
        return r.codigo
    raise ValueError(accion)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
