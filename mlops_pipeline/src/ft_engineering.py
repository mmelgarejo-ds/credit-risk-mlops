"""
Ingeniería de características para el modelo de riesgo crediticio.

Convierte las decisiones documentadas en comprension_eda.ipynb en funciones
reutilizables, de modo que el mismo tratamiento se aplique al entrenar el
modelo y al procesar solicitudes nuevas.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Columnas eliminadas por fuga de información (ver EDA, sección de limpieza)
COLUMNAS_FUGA = ["puntaje", "saldo_mora", "saldo_mora_codeudor"]

# Umbrales de validación definidos en las conclusiones del EDA
SALARIO_MINIMO = 100_000
EDAD_MAXIMA = 90
PUNTAJE_MINIMO = 150
PUNTAJE_MAXIMO = 950


def cargar_datos(ruta=None):
    """Lee el dataset desde el CSV generado por Cargar_datos.ipynb."""
    if ruta is None:
        ruta = Path(__file__).resolve().parents[2] / "Base_de_datos.csv"
    return pd.read_csv(ruta)


def limpiar_datos(df):
    """Aplica las correcciones documentadas en el EDA.

    Los valores inválidos se convierten a nulo en lugar de eliminar la fila:
    el resto de las variables del registro sigue siendo válido.
    """
    df = df.copy()

    # La fecha llega como texto desde el CSV
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    # El dataset usa el 0 como sustituto de dato faltante
    df.loc[df["salario_cliente"] < SALARIO_MINIMO, "salario_cliente"] = np.nan
    df.loc[~df["puntaje_datacredito"].between(PUNTAJE_MINIMO, PUNTAJE_MAXIMO),
           "puntaje_datacredito"] = np.nan

    # Edad implausible para un titular de crédito activo
    df.loc[df["edad_cliente"] > EDAD_MAXIMA, "edad_cliente"] = np.nan

    # Los valores numéricos en tendencia_ingresos quedan como categoría propia
    es_numerico = pd.to_numeric(df["tendencia_ingresos"], errors="coerce").notna()
    df.loc[es_numerico, "tendencia_ingresos"] = "valor_anomalo"

    # Variables con fuga: no estarían disponibles al evaluar una solicitud nueva
    df = df.drop(columns=[c for c in COLUMNAS_FUGA if c in df.columns])

    return df


def crear_atributos(df):
    """Genera las variables derivadas identificadas en el EDA."""
    df = df.copy()

    # Indicadores de ausencia: el hecho de que falte el dato es informativo.
    # Los registros sin datos de Datacrédito presentan mayor tasa de impago
    # (5,73% vs 4,38%, chi-cuadrado p=0,0038).
    df["sin_datos_datacredito"] = df["promedio_ingresos_datacredito"].isna().astype(int)
    df["sin_otros_prestamos"] = (df["total_otros_prestamos"] == 0).astype(int)
    df["sin_saldo"] = (df["saldo_total"] == 0).astype(int)

    # Ratios de capacidad de pago: métricas estándar en riesgo crediticio.
    df["ratio_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"]
    df["ratio_capital_salario"] = df["capital_prestado"] / df["salario_cliente"]
    df["ratio_deuda_salario"] = df["total_otros_prestamos"] / df["salario_cliente"]

    # Componente de la deuda que no es capital. Ambos saldos coinciden en el
    # 83,6% de los registros; la diferencia distingue a los clientes con
    # conceptos adicionales acumulados.
    df["saldo_no_capital"] = df["saldo_total"] - df["saldo_principal"]

    # Atributos temporales: la fecha completa funciona como identificador
    # (10.758 valores únicos sobre 10.763 registros).
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["trimestre_prestamo"] = df["fecha_prestamo"].dt.quarter

    return df


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

# Las variables de monto tienen sesgo fuerte (skew de 3,7 a 43,7).
# El logaritmo lo corrige: capital_prestado pasa de 3,72 a 0,11.
COLS_MONTO = [
    "capital_prestado", "salario_cliente", "total_otros_prestamos",
    "cuota_pactada", "promedio_ingresos_datacredito",
    "saldo_total", "saldo_principal", "saldo_no_capital"
]

COLS_NUMERICAS = [
    "edad_cliente", "plazo_meses", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "ratio_cuota_salario", "ratio_capital_salario", "ratio_deuda_salario",
    "mes_prestamo", "trimestre_prestamo",
    "sin_datos_datacredito", "sin_otros_prestamos", "sin_saldo"
]

COLS_NOMINALES = ["tipo_laboral", "tipo_credito"]

# tendencia_ingresos tiene orden real: la tasa de impago desciende de
# Decreciente (6,27%) a Estable (4,63%) a Creciente (3,91%).
COLS_ORDINALES = ["tendencia_ingresos"]
ORDEN_TENDENCIA = [["Decreciente", "valor_anomalo", "Estable", "Creciente"]]


def construir_preprocesador():
    """Arma el ColumnTransformer con una ruta por tipo de variable."""

    ruta_monto = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler())
    ])

    ruta_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler())
    ])

    ruta_nominal = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ruta_ordinal = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OrdinalEncoder(categories=ORDEN_TENDENCIA,
                                     handle_unknown="use_encoded_value",
                                     unknown_value=-1))
    ])

    return ColumnTransformer([
        ("monto", ruta_monto, COLS_MONTO),
        ("num", ruta_numerica, COLS_NUMERICAS),
        ("nom", ruta_nominal, COLS_NOMINALES),
        ("ord", ruta_ordinal, COLS_ORDINALES)
    ])

if __name__ == "__main__":
    # Este bloque solo se ejecuta al correr el archivo directamente.
    # Si otro módulo importa estas funciones, no se ejecuta nada.

    df = cargar_datos()
    print(f"Datos crudos:        {df.shape[0]} filas, {df.shape[1]} columnas")

    df = limpiar_datos(df)
    print(f"Tras limpieza:       {df.shape[1]} columnas "
          f"({df.isna().sum().sum()} valores marcados como faltantes)")

    df = crear_atributos(df)
    print(f"Con derivados:       {df.shape[1]} columnas")

    # Verificación del preprocesador sobre una partición de prueba
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Pago_atiempo", "fecha_prestamo"])
    y = df["Pago_atiempo"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    prep = construir_preprocesador()
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    print(f"\nTrain preprocesado:  {X_train_t.shape}")
    print(f"Test preprocesado:   {X_test_t.shape}")
    print(f"Nulos restantes:     {np.isnan(X_train_t).sum()}")