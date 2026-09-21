"""
API de inferencia para el modelo de riesgo crediticio.

Expone el pipeline entrenado como servicio HTTP, de modo que otros sistemas
puedan solicitar una evaluación enviando los datos de un cliente.

Ejecución:  uvicorn model_deploy:app --reload
Documentación interactiva:  http://localhost:8000/docs
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RUTA_MODELO = Path(__file__).resolve().parents[2] / "modelo_final.pkl"

app = FastAPI(
    title="Riesgo Crediticio",
    description="Predice la probabilidad de impago de una solicitud de crédito.",
    version="1.2.0"
)

modelo = None


@app.on_event("startup")
def cargar_modelo():
    """Carga el pipeline una sola vez, al arrancar el servicio."""
    global modelo
    if RUTA_MODELO.exists():
        modelo = joblib.load(RUTA_MODELO)


class Solicitud(BaseModel):
    """Estructura esperada en el cuerpo de la petición.

    Pydantic valida tipos y rangos antes de que los datos lleguen al modelo:
    una solicitud con edad 150 se rechaza sin llegar a procesarse.
    """
    tipo_credito: int = Field(..., ge=1, le=99, examples=[4])
    capital_prestado: float = Field(..., gt=0, examples=[2_000_000])
    plazo_meses: int = Field(..., ge=1, le=36, examples=[12])
    edad_cliente: float = Field(..., ge=18, le=90, examples=[42])
    tipo_laboral: str = Field(..., examples=["Empleado"])
    salario_cliente: float = Field(..., gt=100_000, examples=[3_000_000])
    total_otros_prestamos: float = Field(..., ge=0, examples=[1_000_000])
    cuota_pactada: float = Field(..., gt=0, examples=[180_000])
    puntaje_datacredito: float = Field(..., ge=150, le=950, examples=[780])
    cant_creditosvigentes: int = Field(..., ge=0, examples=[5])
    huella_consulta: int = Field(..., ge=0, examples=[4])
    saldo_total: float = Field(..., ge=0, examples=[16_000])
    saldo_principal: float = Field(..., ge=0, examples=[14_000])
    creditos_sectorFinanciero: int = Field(..., ge=0, examples=[2])
    creditos_sectorCooperativo: int = Field(..., ge=0, examples=[0])
    creditos_sectorReal: int = Field(..., ge=0, examples=[1])
    # Opcional: su ausencia es informativa. Omitirlo equivale al dato faltante
    # que ft_engineering marca con isna(), no a un valor de cero.
    promedio_ingresos_datacredito: float | None = Field(None, ge=0, examples=[1_200_000])
    tendencia_ingresos: str = Field(..., examples=["Creciente"])
    mes_prestamo: int = Field(6, ge=1, le=12)
    trimestre_prestamo: int = Field(2, ge=1, le=4)


class Respuesta(BaseModel):
    """Estructura de la respuesta."""
    probabilidad_impago: float
    clasificacion: str
    riesgo_alto: bool

@app.get("/")
def raiz():
    """Información básica del servicio."""
    return {
        "servicio": "Riesgo Crediticio",
        "version": "1.2.0",
        "modelo_cargado": modelo is not None,
        "documentacion": "/docs"
    }


@app.get("/health")
def estado():
    """Verifica que el servicio esté operativo y el modelo disponible.

    Los orquestadores de contenedores consultan este endpoint para saber si
    la instancia está en condiciones de recibir tráfico.
    """
    if modelo is None:
        raise HTTPException(status_code=503,
                            detail="Modelo no disponible")
    return {"estado": "ok"}


@app.post("/predecir", response_model=Respuesta)
def predecir(solicitud: Solicitud):
    """Evalúa una solicitud y devuelve la probabilidad de impago."""
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecutar model_training_evaluation.py"
        )

    datos = solicitud.model_dump()

    # Los atributos derivados se calculan con las mismas fórmulas que
    # ft_engineering.crear_atributos()
    # Mismo criterio que ft_engineering.crear_atributos(): la marca señala
    # ausencia del dato, no un valor de cero.
    sin_dc = datos["promedio_ingresos_datacredito"] is None
    datos["sin_datos_datacredito"] = int(sin_dc)
    if sin_dc:
        # El pipeline imputa el faltante con la mediana del entrenamiento.
        datos["promedio_ingresos_datacredito"] = float("nan")
    datos["sin_otros_prestamos"] = int(datos["total_otros_prestamos"] == 0)
    datos["sin_saldo"] = int(datos["saldo_total"] == 0)
    datos["ratio_cuota_salario"] = datos["cuota_pactada"] / datos["salario_cliente"]
    datos["ratio_capital_salario"] = datos["capital_prestado"] / datos["salario_cliente"]
    datos["ratio_deuda_salario"] = datos["total_otros_prestamos"] / datos["salario_cliente"]
    datos["saldo_no_capital"] = datos["saldo_total"] - datos["saldo_principal"]

    df = pd.DataFrame([datos])

    try:
        prob_impago = float(modelo.predict_proba(df)[0, 0])
        prediccion = int(modelo.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error al procesar la solicitud: {e}")

    return Respuesta(
        probabilidad_impago=round(prob_impago, 4),
        clasificacion="riesgo alto" if prediccion == 0 else "riesgo bajo",
        riesgo_alto=(prediccion == 0)
    )