"""Tests de la API de inferencia.

Se usa TestClient de FastAPI, que ejecuta la aplicación en memoria: no hace
falta levantar el servidor con uvicorn.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_deploy import app  # noqa: E402

# Solicitud de referencia: un cliente con perfil promedio.
SOLICITUD = {
    "tipo_credito": 4,
    "capital_prestado": 2_000_000,
    "plazo_meses": 12,
    "edad_cliente": 42,
    "tipo_laboral": "Empleado",
    "salario_cliente": 3_000_000,
    "total_otros_prestamos": 1_000_000,
    "cuota_pactada": 180_000,
    "puntaje_datacredito": 780,
    "cant_creditosvigentes": 5,
    "huella_consulta": 4,
    "saldo_total": 16_000,
    "saldo_principal": 14_000,
    "creditos_sectorFinanciero": 2,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 1,
    "promedio_ingresos_datacredito": 1_200_000,
    "tendencia_ingresos": "Creciente",
}


@pytest.fixture(scope="module")
def cliente():
    """TestClient como context manager: dispara el evento de startup que
    carga el modelo."""
    with TestClient(app) as c:
        yield c


# --- Endpoints informativos ------------------------------------------------

def test_raiz_responde(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert respuesta.json()["modelo_cargado"] is True


def test_health_indica_servicio_operativo(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"


# --- Predicción ------------------------------------------------------------

def test_prediccion_devuelve_los_tres_campos(cliente):
    respuesta = cliente.post("/predecir", json=SOLICITUD)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"probabilidad_impago", "clasificacion", "riesgo_alto"}


def test_probabilidad_esta_entre_cero_y_uno(cliente):
    cuerpo = cliente.post("/predecir", json=SOLICITUD).json()
    assert 0.0 <= cuerpo["probabilidad_impago"] <= 1.0


def test_clasificacion_es_coherente_con_la_bandera(cliente):
    cuerpo = cliente.post("/predecir", json=SOLICITUD).json()
    esperado = "riesgo alto" if cuerpo["riesgo_alto"] else "riesgo bajo"
    assert cuerpo["clasificacion"] == esperado


# --- Validación de entrada -------------------------------------------------

def test_edad_fuera_de_rango_se_rechaza(cliente):
    """La regla 18-90 definida en el EDA se aplica antes del modelo."""
    invalida = {**SOLICITUD, "edad_cliente": 150}
    assert cliente.post("/predecir", json=invalida).status_code == 422


def test_salario_por_debajo_del_minimo_se_rechaza(cliente):
    invalida = {**SOLICITUD, "salario_cliente": 1_000}
    assert cliente.post("/predecir", json=invalida).status_code == 422


def test_puntaje_fuera_de_rango_se_rechaza(cliente):
    invalida = {**SOLICITUD, "puntaje_datacredito": 1_500}
    assert cliente.post("/predecir", json=invalida).status_code == 422


def test_campo_obligatorio_faltante_se_rechaza(cliente):
    incompleta = {k: v for k, v in SOLICITUD.items() if k != "salario_cliente"}
    assert cliente.post("/predecir", json=incompleta).status_code == 422


# --- Dato faltante de Datacrédito ------------------------------------------

def test_ingresos_de_datacredito_son_opcionales(cliente):
    """Omitir el dato es válido: su ausencia es informativa, no un error."""
    sin_dato = {k: v for k, v in SOLICITUD.items()
                if k != "promedio_ingresos_datacredito"}
    respuesta = cliente.post("/predecir", json=sin_dato)
    assert respuesta.status_code == 200
    assert 0.0 <= respuesta.json()["probabilidad_impago"] <= 1.0
