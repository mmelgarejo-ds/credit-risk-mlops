"""Tests del pipeline de preparación de datos.

Verifican que las reglas de validación y los atributos derivados definidos en
comprension_eda.ipynb se apliquen efectivamente sobre cualquier conjunto de
datos que entre al pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Los tests viven en mlops_pipeline/tests y el código en mlops_pipeline/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ft_engineering import (  # noqa: E402
    COLUMNAS_FUGA,
    EDAD_MAXIMA,
    PUNTAJE_MAXIMO,
    PUNTAJE_MINIMO,
    SALARIO_MINIMO,
    cargar_datos,
    construir_preprocesador,
    crear_atributos,
    limpiar_datos,
)


@pytest.fixture(scope="module")
def df_crudo():
    """Dataset original, leído una sola vez para todos los tests."""
    return cargar_datos()


@pytest.fixture(scope="module")
def df_limpio(df_crudo):
    return limpiar_datos(df_crudo)


@pytest.fixture(scope="module")
def df_completo(df_limpio):
    return crear_atributos(df_limpio)


# --- Carga -----------------------------------------------------------------

def test_cargar_datos_devuelve_dataframe_no_vacio(df_crudo):
    assert isinstance(df_crudo, pd.DataFrame)
    assert len(df_crudo) > 0
    assert "Pago_atiempo" in df_crudo.columns


# --- Limpieza --------------------------------------------------------------

def test_limpieza_no_elimina_filas(df_crudo, df_limpio):
    """Los valores inválidos se marcan como nulos, no se descartan registros."""
    assert len(df_limpio) == len(df_crudo)


def test_elimina_columnas_con_fuga(df_limpio):
    for columna in COLUMNAS_FUGA:
        assert columna not in df_limpio.columns


def test_fecha_convertida_a_datetime(df_limpio):
    assert pd.api.types.is_datetime64_any_dtype(df_limpio["fecha_prestamo"])


def test_salarios_por_debajo_del_minimo_quedan_nulos(df_limpio):
    """El dataset usa el 0 y montos irrisorios como dato faltante."""
    salarios = df_limpio["salario_cliente"].dropna()
    assert (salarios >= SALARIO_MINIMO).all()


def test_edades_implausibles_quedan_nulas(df_limpio):
    edades = df_limpio["edad_cliente"].dropna()
    assert (edades <= EDAD_MAXIMA).all()


def test_puntaje_datacredito_dentro_del_rango_valido(df_limpio):
    puntajes = df_limpio["puntaje_datacredito"].dropna()
    assert puntajes.between(PUNTAJE_MINIMO, PUNTAJE_MAXIMO).all()


def test_tendencia_ingresos_sin_valores_numericos(df_limpio):
    """Los 58 registros numéricos se agrupan bajo una categoría propia."""
    categorias = set(df_limpio["tendencia_ingresos"].dropna().unique())
    assert categorias <= {"Creciente", "Estable", "Decreciente", "valor_anomalo"}


# --- Atributos derivados ---------------------------------------------------

def test_crear_atributos_agrega_las_nueve_columnas(df_limpio, df_completo):
    nuevas = [
        "sin_datos_datacredito", "sin_otros_prestamos", "sin_saldo",
        "ratio_cuota_salario", "ratio_capital_salario", "ratio_deuda_salario",
        "saldo_no_capital", "mes_prestamo", "trimestre_prestamo",
    ]
    for columna in nuevas:
        assert columna in df_completo.columns
    assert df_completo.shape[1] == df_limpio.shape[1] + len(nuevas)


def test_indicadores_de_ausencia_son_binarios(df_completo):
    for columna in ["sin_datos_datacredito", "sin_otros_prestamos", "sin_saldo"]:
        assert set(df_completo[columna].unique()) <= {0, 1}


def test_indicador_marca_exactamente_los_faltantes(df_completo):
    """La marca tiene que coincidir con los nulos de la variable de origen."""
    esperado = df_completo["promedio_ingresos_datacredito"].isna().astype(int)
    assert (df_completo["sin_datos_datacredito"] == esperado).all()


def test_ratio_cuota_salario_se_calcula_correctamente(df_completo):
    fila = df_completo.dropna(subset=["cuota_pactada", "salario_cliente"]).iloc[0]
    assert fila["ratio_cuota_salario"] == pytest.approx(
        fila["cuota_pactada"] / fila["salario_cliente"])


def test_saldo_no_capital_es_la_diferencia_entre_saldos(df_completo):
    sub = df_completo.dropna(subset=["saldo_total", "saldo_principal"])
    esperado = sub["saldo_total"] - sub["saldo_principal"]
    assert np.allclose(sub["saldo_no_capital"], esperado)


def test_atributos_temporales_dentro_de_rango(df_completo):
    assert df_completo["mes_prestamo"].between(1, 12).all()
    assert df_completo["trimestre_prestamo"].between(1, 4).all()


# --- Preprocesador ---------------------------------------------------------

def test_preprocesador_no_deja_valores_nulos(df_completo):
    """Tras el preprocesamiento el modelo no puede recibir vacíos."""
    X = df_completo.drop(columns=["Pago_atiempo", "fecha_prestamo"])
    transformado = construir_preprocesador().fit_transform(X)
    assert not np.isnan(transformado).any()


def test_preprocesador_ajustado_en_train_se_aplica_a_test(df_completo):
    """fit_transform en entrenamiento, transform en prueba: mismo ancho."""
    X = df_completo.drop(columns=["Pago_atiempo", "fecha_prestamo"])
    corte = int(len(X) * 0.8)
    prep = construir_preprocesador()
    train = prep.fit_transform(X.iloc[:corte])
    test = prep.transform(X.iloc[corte:])
    assert train.shape[1] == test.shape[1]
