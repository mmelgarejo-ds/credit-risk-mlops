"""Tests de las funciones de entrenamiento y evaluación.

No se testea el entrenamiento en sí (tarda minutos por el GridSearch): se
prueban las funciones auxiliares con datos simulados pequeños.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_training_evaluation import (  # noqa: E402
    calcular_metricas,
    construir_modelos,
    optimizar_umbral,
)


# --- calcular_metricas ------------------------------------------------------

def test_calcular_metricas_devuelve_las_cuatro_claves():
    y_real = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.8, 0.1, 0.6, 0.7, 0.2])  # probabilidad de impago
    metricas = calcular_metricas(y_real, y_pred, y_proba)
    assert set(metricas) == {"precision", "recall", "f1", "roc_auc"}


def test_calcular_metricas_prediccion_perfecta_da_recall_uno():
    """Si el modelo acierta todos los impagos, el recall sobre la clase 0
    tiene que ser 1.0.
    """
    y_real = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])
    metricas = calcular_metricas(y_real, y_pred, y_proba)
    assert metricas["recall"] == pytest.approx(1.0)


def test_calcular_metricas_valores_entre_cero_y_uno():
    y_real = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.7, 0.3, 0.6, 0.2, 0.8, 0.4])
    metricas = calcular_metricas(y_real, y_pred, y_proba)
    for valor in metricas.values():
        assert 0.0 <= valor <= 1.0


# --- construir_modelos ------------------------------------------------------

def test_construir_modelos_devuelve_los_tres_esperados():
    modelos = construir_modelos()
    assert set(modelos) == {"LogisticRegression", "RandomForest", "XGBoost"}


def test_construir_modelos_calcula_razon_desde_los_datos():
    """La razón de balanceo de XGBoost debe reflejar la proporción real de
    clases del conjunto de entrenamiento recibido, no un valor fijo.
    """
    y_entrenamiento = np.array([1] * 80 + [0] * 20)  # 80 pagos, 20 impagos
    modelos = construir_modelos(y_entrenamiento)
    razon = modelos["XGBoost"].get_params()["scale_pos_weight"]
    assert razon == pytest.approx(80 / 20)


def test_construir_modelos_sin_datos_usa_valor_por_defecto():
    modelos = construir_modelos(y_entrenamiento=None)
    razon = modelos["XGBoost"].get_params()["scale_pos_weight"]
    assert razon == pytest.approx(20.0)


# --- optimizar_umbral --------------------------------------------------------

class ModeloFalso:
    """Simula un pipeline entrenado: predict_proba fijo, sin entrenar nada."""
    def __init__(self, probabilidades_impago):
        self._proba = np.asarray(probabilidades_impago)

    def predict_proba(self, X):
        return np.column_stack([self._proba, 1 - self._proba])


def test_optimizar_umbral_devuelve_una_fila_por_umbral():
    y_test = np.array([0, 0, 1, 1, 1])
    proba_impago = np.array([0.9, 0.6, 0.4, 0.2, 0.1])
    modelo = ModeloFalso(proba_impago)
    tabla = optimizar_umbral(modelo, X_test=None, y_test=y_test,
                             umbrales=np.array([0.3, 0.5, 0.7]))
    assert len(tabla) == 3


def test_optimizar_umbral_mas_detectados_a_menor_umbral():
    """Bajar el umbral no puede reducir la cantidad de impagos detectados."""
    y_test = np.array([0, 0, 0, 1, 1])
    proba_impago = np.array([0.9, 0.6, 0.3, 0.2, 0.1])
    modelo = ModeloFalso(proba_impago)
    tabla = optimizar_umbral(modelo, X_test=None, y_test=y_test,
                             umbrales=np.array([0.2, 0.8]))
    detectados_umbral_bajo = tabla.loc[0.2, "detectados"]
    detectados_umbral_alto = tabla.loc[0.8, "detectados"]
    assert detectados_umbral_bajo >= detectados_umbral_alto