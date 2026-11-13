"""Códigos de salida de ``entrenar_mlflow`` y ``arranque`` (etapa 6). Módulo sin
dependencias: los comandos verifican la procedencia del código **antes** de importar el
resto del proyecto (un archivo modificado no llega a ejecutarse)."""

OK, ERROR, PROCEDENCIA, SIN_MODELO, INDETERMINADA, OCUPADO, PERDIDA = 0, 1, 2, 3, 4, 5, 6
