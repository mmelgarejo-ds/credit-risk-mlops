"""Tests del módulo de detección de drift."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_monitoring import calcular_psi, detectar_drift_numerico  # noqa: E402


def test_psi_es_cero_cuando_las_distribuciones_son_iguales():
    """Comparar una muestra contra sí misma no debería mostrar cambio."""
    datos = pd.DataFrame({"x": np.random.default_rng(42).normal(0, 1, 1000)})
    psi = calcular_psi(datos, datos, "x")
    assert psi == pytest.approx(0, abs=0.01)


def test_psi_detecta_un_cambio_de_media():
    """Un desplazamiento grande en la media debe producir PSI alto."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    act = pd.DataFrame({"x": rng.normal(5, 1, 1000)})
    psi = calcular_psi(ref, act, "x")
    assert psi > 0.25


def test_drift_numerico_no_falla_con_columnas_de_pocos_datos():
    """Las columnas con menos de 30 observaciones se excluyen del resultado,
    no deben generar error.
    """
    ref = pd.DataFrame({
        "x": range(50),
        "y": list(range(10)) + [None] * 40,
    })
    act = pd.DataFrame({
        "x": range(50, 100),
        "y": list(range(10, 20)) + [None] * 40,
    })
    resultado = detectar_drift_numerico(ref, act, ["x", "y"])
    assert "x" in resultado["variable"].values
    assert "y" not in resultado["variable"].values


def test_drift_numerico_incluye_columna_psi():
    ref = pd.DataFrame({"x": range(100)})
    act = pd.DataFrame({"x": range(100, 200)})
    resultado = detectar_drift_numerico(ref, act, ["x"])
    assert "psi" in resultado.columns