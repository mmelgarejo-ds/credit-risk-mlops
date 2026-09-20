"""
Aplicación Streamlit para el modelo de riesgo crediticio.

Permite evaluar solicitudes individuales sin necesidad de escribir código, y
consultar el estado del monitoreo de drift. Está pensada para usuarios del
área de Riesgos.

Ejecución:  streamlit run app_streamlit.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from ft_engineering import (cargar_datos, limpiar_datos, crear_atributos,
                            construir_preprocesador)

RUTA_MODELO = Path(__file__).resolve().parents[2] / "modelo_final.pkl"

st.set_page_config(page_title="Riesgo Crediticio", layout="wide")


@st.cache_resource
def cargar_modelo():
    """Carga el pipeline entrenado.

    @st.cache_resource evita releerlo en cada interacción del usuario.
    """
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_dataset():
    """Carga los datos procesados, usados como referencia y para el drift."""
    return crear_atributos(limpiar_datos(cargar_datos()))


modelo = cargar_modelo()
df = cargar_dataset()

st.title("Modelo de riesgo crediticio")
st.caption("Proyecto Integrador M5 — Henry")

if modelo is None:
    st.error(
        "No se encontró `modelo_final.pkl`. "
        "Ejecutá primero `python model_training_evaluation.py`."
    )
    st.stop()
    
tab_pred, tab_drift = st.tabs(["Evaluar solicitud", "Monitoreo de drift"])

with tab_pred:
    st.subheader("Datos de la solicitud")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad del cliente", 18, 90, 42)
        tipo_laboral = st.selectbox("Situación laboral",
                                    ["Empleado", "Independiente"])
        salario = st.number_input("Salario mensual", 100_000, 50_000_000,
                                  3_000_000, step=100_000)
        tendencia = st.selectbox("Tendencia de ingresos",
                                 ["Creciente", "Estable", "Decreciente"])

    with col2:
        capital = st.number_input("Capital solicitado", 300_000, 50_000_000,
                                  2_000_000, step=100_000)
        plazo = st.selectbox("Plazo (meses)", [6, 10, 12, 18, 24, 36], index=2)
        cuota = st.number_input("Cuota pactada", 20_000, 4_000_000,
                                180_000, step=10_000)
        tipo_credito = st.selectbox("Tipo de crédito", [4, 9, 10, 6, 7])

    with col3:
        puntaje_dc = st.number_input("Puntaje Datacrédito", 150, 950, 780)
        creditos_vigentes = st.number_input("Créditos vigentes", 0, 60, 5)
        consultas = st.number_input("Consultas al historial", 0, 30, 4)
        otros_prestamos = st.number_input("Otros préstamos", 0, 100_000_000,
                                          1_000_000, step=100_000)

    with st.expander("Variables adicionales"):
        c1, c2, c3 = st.columns(3)
        saldo_total = c1.number_input("Saldo total", 0, 5_000_000, 16_000)
        saldo_principal = c2.number_input("Saldo principal", 0, 2_000_000, 14_000)
        ingresos_dc = c3.number_input("Ingresos reportados (Datacrédito)",
                                      0, 40_000_000, 1_200_000, step=100_000)
        c4, c5, c6 = st.columns(3)
        sec_financiero = c4.number_input("Créditos sector financiero", 0, 51, 2)
        sec_cooperativo = c5.number_input("Créditos sector cooperativo", 0, 13, 0)
        sec_real = c6.number_input("Créditos sector real", 0, 25, 1)

        st.divider()

    if st.button("Evaluar riesgo", type="primary", use_container_width=True):
        # Se arma un DataFrame de una fila con la estructura que espera el
        # pipeline. Los nombres deben coincidir con los del entrenamiento.
        solicitud = pd.DataFrame([{
            "tipo_credito": tipo_credito,
            "capital_prestado": float(capital),
            "plazo_meses": plazo,
            "edad_cliente": float(edad),
            "tipo_laboral": tipo_laboral,
            "salario_cliente": float(salario),
            "total_otros_prestamos": otros_prestamos,
            "cuota_pactada": cuota,
            "puntaje_datacredito": float(puntaje_dc),
            "cant_creditosvigentes": creditos_vigentes,
            "huella_consulta": consultas,
            "saldo_total": float(saldo_total),
            "saldo_principal": float(saldo_principal),
            "creditos_sectorFinanciero": sec_financiero,
            "creditos_sectorCooperativo": sec_cooperativo,
            "creditos_sectorReal": sec_real,
            "promedio_ingresos_datacredito": float(ingresos_dc),
            "tendencia_ingresos": tendencia,
            # Atributos derivados, calculados igual que en ft_engineering
            "sin_datos_datacredito": 0,
            "sin_otros_prestamos": int(otros_prestamos == 0),
            "sin_saldo": int(saldo_total == 0),
            "ratio_cuota_salario": cuota / salario,
            "ratio_capital_salario": capital / salario,
            "ratio_deuda_salario": otros_prestamos / salario,
            "saldo_no_capital": saldo_total - saldo_principal,
            "mes_prestamo": 6,
            "trimestre_prestamo": 2,
        }])

        prob_impago = modelo.predict_proba(solicitud)[0, 0]
        prediccion = modelo.predict(solicitud)[0]

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.metric("Probabilidad de impago", f"{prob_impago:.1%}")

        with col_b:
            if prediccion == 0:
                st.error("**Riesgo alto** — se sugiere revisión manual")
            else:
                st.success("**Riesgo bajo** — perfil compatible con aprobación")

        st.progress(float(prob_impago))

        st.caption(
            "El modelo clasifica como riesgo alto cuando la probabilidad de "
            "impago supera el 50%. Ese umbral es ajustable: bajarlo detecta "
            "más impagos a costa de más falsas alarmas."
        )
with tab_drift:
    st.subheader("Monitoreo de data drift")
    st.write(
        "Compara la distribución de las variables entre dos particiones del "
        "dataset. La partición aleatoria funciona como control: si detectara "
        "drift, el problema estaría en el método y no en los datos."
    )

    from model_monitoring import (experimento_control, experimento_temporal)

    columnas_cat = ["tipo_laboral", "tendencia_ingresos"]
    excluidas = ["Pago_atiempo", "mes_prestamo", "trimestre_prestamo"]
    columnas_num = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in excluidas]

    if st.button("Ejecutar análisis de drift"):
        with st.spinner("Calculando..."):
            num_ctrl, cat_ctrl = experimento_control(df, columnas_num, columnas_cat)
            num_temp, cat_temp = experimento_temporal(df, columnas_num, columnas_cat)

        total = len(num_ctrl) + len(cat_ctrl)
        drift_ctrl = int(num_ctrl["drift"].sum() + cat_ctrl["drift"].sum())
        drift_temp = int(num_temp["drift"].sum() + cat_temp["drift"].sum())

        c1, c2 = st.columns(2)
        c1.metric("Partición aleatoria (control)", f"{drift_ctrl} / {total}")
        c2.metric("Partición cronológica", f"{drift_temp} / {total}")

        if drift_temp > drift_ctrl:
            st.warning(
                f"Se detecta drift temporal en {drift_temp} de {total} variables. "
                f"El control arroja {drift_ctrl}, lo que descarta un artefacto "
                "del método de detección."
            )

        st.write("**Variables ordenadas por estadístico KS (partición temporal)**")
        st.dataframe(num_temp, use_container_width=True)

        st.write("**Variables categóricas (chi-cuadrado)**")
        st.dataframe(cat_temp, use_container_width=True)