"""
Monitoreo de data drift para el modelo de riesgo crediticio.

Compara la distribución de las variables entre dos conjuntos de datos para
detectar si los datos de producción se apartan de los de entrenamiento. Un
modelo entrenado sobre una distribución deja de ser confiable cuando la
distribución cambia, sin que el sistema emita ningún error.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

from ft_engineering import cargar_datos, limpiar_datos, crear_atributos

SEMILLA = 42

# Umbrales del estadístico KS (ver lecture 3):
#   < 0,15  sin drift significativo
#   0,15-0,25  drift moderado, investigar
#   > 0,25  drift significativo, evaluar reentrenamiento
UMBRAL_KS = 0.25
UMBRAL_P = 0.05
UMBRAL_PSI = 0.25

def calcular_psi(ref, act, columna, bins=10):
    """Population Stability Index: mide cuánto cambió la distribución de una
    variable entre dos períodos, dividiéndola en bins y comparando qué
    proporción de cada muestra cae en cada uno.

    Umbrales estándar en la industria:
        < 0,10        sin cambio relevante
        0,10 - 0,25   cambio moderado
        > 0,25        cambio significativo
    """
    a = ref[columna].dropna()
    b = act[columna].dropna()

    # Los puntos de corte se definen sobre la referencia, no sobre el actual:
    # el PSI mide qué tanto se aparta "act" de la distribución "normal".
    cortes = np.percentile(a, np.linspace(0, 100, bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)  # evita bins vacíos si hay muchos valores repetidos

    prop_ref = pd.cut(a, bins=cortes).value_counts(normalize=True, sort=False)
    prop_act = pd.cut(b, bins=cortes).value_counts(normalize=True, sort=False)

    # Evita log(0): un bin sin observaciones se reemplaza por un valor mínimo
    prop_ref = prop_ref.replace(0, 0.0001)
    prop_act = prop_act.replace(0, 0.0001)

    return float(((prop_act - prop_ref) * np.log(prop_act / prop_ref)).sum())

def detectar_drift_numerico(ref, act, columnas):
    """Aplica KS y PSI a cada variable numérica.

    KS compara las distribuciones acumuladas y no asume normalidad, necesario
    aquí porque varias variables presentan asimetría superior a 20. PSI es el
    estándar de la industria para monitoreo de scorecards y complementa a KS:
    dos métricas independientes reducen el riesgo de que un falso positivo de
    una se tome como drift real.
    """
    filas = []
    for col in columnas:
        a = ref[col].dropna()
        b = act[col].dropna()
        if len(a) < 30 or len(b) < 30:
            continue

        estadistico, p_valor = ks_2samp(a, b)
        psi = calcular_psi(ref, act, col)

        filas.append({
            "variable": col,
            "ks": round(estadistico, 4),
            "p_valor": round(p_valor, 6),
            "psi": round(psi, 4),
            "drift": estadistico > UMBRAL_KS or psi > UMBRAL_PSI
        })

    return pd.DataFrame(filas).sort_values("ks", ascending=False)


def detectar_drift_categorico(ref, act, columnas):
    """Aplica chi-cuadrado a cada variable categórica.

    Compara la frecuencia de cada categoría entre ambos conjuntos.
    """
    filas = []
    for col in columnas:
        tabla = pd.crosstab(
            pd.concat([ref[col], act[col]]),
            ["referencia"] * len(ref) + ["actual"] * len(act)
        )
        if tabla.shape[0] < 2:
            continue

        chi2, p_valor, _, _ = chi2_contingency(tabla)
        filas.append({
            "variable": col,
            "chi2": round(chi2, 4),
            "p_valor": round(p_valor, 6),
            "drift": p_valor < UMBRAL_P
        })

    return pd.DataFrame(filas).sort_values("p_valor")

def experimento_control(df, columnas_num, columnas_cat):
    """Partición aleatoria: no debería detectarse drift.

    Sirve como control de la función de detección. Si dos mitades tomadas al
    azar del mismo conjunto arrojaran drift, el problema estaría en el método
    y no en los datos.
    """
    barajado = df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    mitad = len(barajado) // 2
    ref, act = barajado.iloc[:mitad], barajado.iloc[mitad:]

    return (detectar_drift_numerico(ref, act, columnas_num),
            detectar_drift_categorico(ref, act, columnas_cat))


def experimento_temporal(df, columnas_num, columnas_cat):
    """Partición cronológica: créditos antiguos contra recientes.

    Es la comparación que reproduce la situación real de producción, donde el
    modelo se entrena con datos históricos y recibe solicitudes nuevas.
    """
    ordenado = df.sort_values("fecha_prestamo").reset_index(drop=True)
    mitad = len(ordenado) // 2
    ref, act = ordenado.iloc[:mitad], ordenado.iloc[mitad:]

    print(f"  Referencia: {ref['fecha_prestamo'].min().date()} a "
          f"{ref['fecha_prestamo'].max().date()}")
    print(f"  Actual:     {act['fecha_prestamo'].min().date()} a "
          f"{act['fecha_prestamo'].max().date()}")

    return (detectar_drift_numerico(ref, act, columnas_num),
            detectar_drift_categorico(ref, act, columnas_cat))

def guardar_reporte(num_temp, cat_temp, drift_ctrl, drift_temp, total):
    """Persiste el resultado de la corrida para que otros sistemas o
    procesos puedan consultarlo sin depender de que el dashboard esté
    abierto.
    """
    import json
    from datetime import datetime, timezone

    carpeta = Path(__file__).resolve().parents[2] / "reports"
    carpeta.mkdir(exist_ok=True)

    ahora = datetime.now(timezone.utc).isoformat()

    if drift_temp == 0:
        nivel = "sin_drift"
    elif drift_temp / total <= 0.15:
        nivel = "moderado"
    else:
        nivel = "critico"

    reporte = {
        "fecha_ejecucion": ahora,
        "nivel_alerta": nivel,
        "variables_con_drift": int(drift_temp),
        "total_variables": int(total),
        "control_variables_con_drift": int(drift_ctrl),
        "detalle_numericas": num_temp.to_dict(orient="records"),
        "detalle_categoricas": cat_temp.to_dict(orient="records"),
    }

    # Reporte de la corrida más reciente: pensado para que lo lea otro
    # sistema o proceso automatizado.
    with open(carpeta / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    # Historial acumulado: permite auditar la evolución sin depender de
    # que el dashboard esté abierto.
    fila_historial = pd.DataFrame([{
        "fecha_ejecucion": ahora,
        "nivel_alerta": nivel,
        "variables_con_drift": drift_temp,
        "total_variables": total,
    }])
    ruta_historial = carpeta / "drift_history.csv"
    fila_historial.to_csv(
        ruta_historial, mode="a", header=not ruta_historial.exists(), index=False
    )

    return carpeta

if __name__ == "__main__":
    df = crear_atributos(limpiar_datos(cargar_datos()))

        # Se excluyen la variable objetivo, la fecha, y los atributos derivados
    # de la fecha: al particionar cronológicamente, mes y trimestre difieren
    # por construcción y no reflejan drift real.
    excluidas = ["Pago_atiempo", "mes_prestamo", "trimestre_prestamo"]
    columnas_num = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in excluidas]
    columnas_cat = ["tipo_laboral", "tendencia_ingresos"]

    print(f"Variables evaluadas: {len(columnas_num)} numéricas, "
          f"{len(columnas_cat)} categóricas\n")

    # --- Control ---
    print("=== Experimento de control (partición aleatoria) ===")
    num_ctrl, cat_ctrl = experimento_control(df, columnas_num, columnas_cat)
    drift_ctrl = num_ctrl["drift"].sum() + cat_ctrl["drift"].sum()
    total = len(num_ctrl) + len(cat_ctrl)
    print(f"Variables con drift: {drift_ctrl} de {total}")
    print(num_ctrl.head(5).to_string(index=False))

    # --- Temporal ---
    print("\n=== Experimento temporal (partición cronológica) ===")
    num_temp, cat_temp = experimento_temporal(df, columnas_num, columnas_cat)
    drift_temp = num_temp["drift"].sum() + cat_temp["drift"].sum()
    print(f"\nVariables con drift: {drift_temp} de {total}")
    print(num_temp.to_string(index=False))

    print("\n--- Variables categóricas ---")
    print(cat_temp.to_string(index=False))

    # --- Conclusión ---
    print("\n=== Resumen ===")
    print(f"Partición aleatoria:   {drift_ctrl}/{total} variables con drift")
    print(f"Partición cronológica: {drift_temp}/{total} variables con drift")
    
    carpeta = guardar_reporte(num_temp, cat_temp, drift_ctrl, drift_temp, total)
    print(f"\nReporte guardado en {carpeta}/")
    
    if drift_temp > drift_ctrl:
        print("\nEl contraste indica drift temporal real: la partición aleatoria")
        print("sirve como control y descarta que el resultado sea un artefacto")
        print("del método de detección.")

