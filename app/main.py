"""Aplicación FastAPI de NovaMercado (contrato: Anexo; diseño: DECISIONES.md §13).

Arranque:  uvicorn app.main:app --port 8000   (variables en .env.example)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.calidad.resultado import SolicitudRechazada
from app.db.conexion import BaseDatosNoDisponible
from app.dependencias import Dependencias, desde_entorno
from app.modelo.proveedores import ModeloNoDisponible
from app.orquestador import OrquestadorNoDisponible
from app.schemas.contrato import EstadoReentrenamiento, EstadoSalud, ModeloVigente, ReentrenamientoAceptado
from app.servicios.observaciones import observar
from app.servicios.prediccion import predecir

logger = logging.getLogger("novamercado.api")

LIMITE_CUERPO_GUARDADO = 2000


class CuerpoNoJSON(SolicitudRechazada):
    def __init__(self, texto: str):
        super().__init__(None, "el cuerpo de la solicitud no es JSON válido")
        self.payload = {"cuerpo_no_json": texto[:LIMITE_CUERPO_GUARDADO]}


async def _leer_json(request: Request) -> object:
    crudo = await request.body()
    texto = crudo.decode("utf-8", errors="replace")
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CuerpoNoJSON(texto) from exc


def create_app(dependencias: Dependencias | None = None, iniciar_hilos: bool = True,
               fabrica: Callable[[], Dependencias] = desde_entorno) -> FastAPI:
    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        deps = dependencias or fabrica()
        app.state.deps = deps
        if iniciar_hilos:
            deps.iniciar()
        try:
            yield
        finally:
            if iniciar_hilos:
                deps.detener()

    app = FastAPI(title="NovaMercado — pronóstico de ventas", version="0.4.0", lifespan=ciclo_de_vida)
    if dependencias is not None:
        app.state.deps = dependencias

    def deps(request: Request) -> Dependencias:
        return request.app.state.deps

    # ------------------------------------------------------------- errores
    @app.exception_handler(SolicitudRechazada)
    async def _rechazo(request: Request, exc: SolicitudRechazada):
        return JSONResponse(status_code=422, content={"detail": exc.detail})

    @app.exception_handler(ModeloNoDisponible)
    async def _sin_modelo(request: Request, exc: ModeloNoDisponible):
        return JSONResponse(status_code=503, content={"detail": f"no hay un modelo disponible: {exc}"})

    @app.exception_handler(BaseDatosNoDisponible)
    async def _sin_bd(request: Request, exc: BaseDatosNoDisponible):
        logger.error("base de datos no disponible durante %s %s", request.method, request.url.path)
        return JSONResponse(status_code=503, content={"detail": "la base de datos no está disponible"})

    @app.exception_handler(RequestValidationError)
    async def _validacion(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "solicitud inválida"})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})

    @app.exception_handler(Exception)
    async def _inesperado(request: Request, exc: Exception):
        logger.exception("error inesperado en %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "error interno del servicio"})

    # ----------------------------------------------------------- endpoints
    @app.post("/v1/prediccion")
    async def v1_prediccion(request: Request):
        d = deps(request)
        try:
            payload = await _leer_json(request)
        except CuerpoNoJSON as exc:
            from app.servicios.rechazos import registrar_rechazo
            await run_in_threadpool(registrar_rechazo, d.persistencia, "prediccion", exc.payload, exc.detail)
            raise
        return await run_in_threadpool(predecir, payload, d.persistencia, d.gestor)

    @app.post("/v1/observaciones")
    async def v1_observaciones(request: Request):
        d = deps(request)
        try:
            payload = await _leer_json(request)
        except CuerpoNoJSON as exc:
            from app.servicios.rechazos import registrar_rechazo
            await run_in_threadpool(registrar_rechazo, d.persistencia, "observacion", exc.payload, exc.detail)
            raise
        return await run_in_threadpool(observar, payload, d.persistencia)

    @app.post("/admin/reentrenar", status_code=202)
    async def admin_reentrenar(request: Request):
        d = deps(request)
        try:
            id_ejecucion = await run_in_threadpool(d.orquestador.iniciar)
        except OrquestadorNoDisponible as exc:
            return JSONResponse(status_code=503, content={"detail": str(exc)})
        return ReentrenamientoAceptado(id=str(id_ejecucion)).model_dump()

    @app.get("/admin/reentrenar/{id_ejecucion}")
    async def admin_estado(id_ejecucion: str, request: Request):
        estado: EstadoReentrenamiento | None = await run_in_threadpool(deps(request).orquestador.estado, id_ejecucion)
        if estado is None:
            return JSONResponse(status_code=404, content={"detail": f"no existe la ejecución {id_ejecucion!r}"})
        return estado.como_respuesta()

    @app.get("/modelo")
    async def modelo(request: Request):
        m = deps(request).gestor.actual()
        return ModeloVigente(nombre=m.nombre, alias=m.alias, version=m.version).model_dump()

    @app.get("/health")
    async def health(request: Request):
        diag = deps(request).monitor.diagnostico()
        cuerpo = EstadoSalud.model_validate(diag.como_respuesta()).model_dump(exclude_none=True)
        return JSONResponse(status_code=diag.codigo_http, content=cuerpo)

    return app


app = create_app()
