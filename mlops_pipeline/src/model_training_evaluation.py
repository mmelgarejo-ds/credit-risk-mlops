"""
Entrenamiento y evaluación de modelos para el riesgo crediticio.

Entrena varios clasificadores, los evalúa con métricas apropiadas para un
dataset desbalanceado (4,75% de impago) y selecciona el de mejor desempeño.

Cobertura de tests ampliada: calcular_metricas, construir_modelos y
optimizar_umbral se validan con datos simulados.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

from ft_engineering import (cargar_datos, limpiar_datos, crear_atributos,
                            construir_preprocesador)

SEMILLA = 42

# La clase de interés es el impago (0), no el pago a tiempo.
# Todas las métricas se calculan sobre ella.
CLASE_POSITIVA = 0


def preparar_datos():
    """Devuelve X e y listos para el split."""
    df = crear_atributos(limpiar_datos(cargar_datos()))
    X = df.drop(columns=["Pago_atiempo", "fecha_prestamo"])
    y = df["Pago_atiempo"]
    return X, y


def calcular_metricas(y_real, y_pred, y_proba):
    """Calcula las métricas sobre la clase minoritaria (impago).

    Accuracy no se incluye: con 95,25% de clase mayoritaria, un modelo que
    prediga 'paga' para todos obtiene 95,25% sin detectar un solo impago.
    """
    return {
        "precision": precision_score(y_real, y_pred, pos_label=CLASE_POSITIVA),
        "recall": recall_score(y_real, y_pred, pos_label=CLASE_POSITIVA),
        "f1": f1_score(y_real, y_pred, pos_label=CLASE_POSITIVA),
        # roc_auc usa la probabilidad, no la predicción binaria.
        # Se pasa la probabilidad de la clase 0.
        "roc_auc": roc_auc_score(y_real, 1 - y_proba)
    }

def construir_modelos(y_entrenamiento=None):
    """Devuelve los clasificadores candidatos.

    El parámetro de balanceo es clave: sin él, los tres modelos tienden a
    predecir siempre la clase mayoritaria, dado que el impago representa
    solo el 4,75% de los registros.

    La razón entre clases se calcula sobre los datos recibidos en lugar de
    fijarse como constante, de modo que siga siendo válida si el dataset
    cambia.
    """
    # scale_pos_weight de XGBoost espera la razón entre clases.
    if y_entrenamiento is not None:
        razon_clases = ((y_entrenamiento == 1).sum()
                        / max((y_entrenamiento == 0).sum(), 1))
    else:
        razon_clases = 20.0

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEMILLA
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEMILLA,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=razon_clases,
            random_state=SEMILLA,
            eval_metric="logloss"
        )
    }

def entrenar_y_evaluar(X_train, X_test, y_train, y_test):
    """Entrena cada modelo y devuelve sus métricas sobre el conjunto de prueba.

    Cada modelo se envuelve junto al preprocesador en un único Pipeline, de
    modo que las transformaciones se ajusten solo con los datos de
    entrenamiento y se apliquen sin recalcular sobre los de prueba.
    """
    resultados = []
    pipelines = {}

    for nombre, modelo in construir_modelos(y_train).items():
        print(f"Entrenando {nombre}...")

        pipe = Pipeline([
            ("preprocesador", construir_preprocesador()),
            ("modelo", modelo)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        # predict_proba devuelve una columna por clase; la 0 es P(impago)
        y_proba = pipe.predict_proba(X_test)[:, 0]

        metricas = calcular_metricas(y_test, y_pred, y_proba)
        metricas["modelo"] = nombre
        resultados.append(metricas)
        pipelines[nombre] = pipe

    tabla = pd.DataFrame(resultados).set_index("modelo")
    return tabla.round(4), pipelines

def optimizar_umbral(pipe, X_test, y_test, umbrales=None):
    """Evalúa distintos umbrales de decisión sobre las probabilidades.

    Por defecto el modelo clasifica como impago cuando P(impago) > 0.5.
    Ese valor es arbitrario: bajarlo aumenta la detección de impagos a
    costa de más falsas alarmas.
    """
    if umbrales is None:
        umbrales = np.arange(0.20, 0.85, 0.05)

    proba_impago = pipe.predict_proba(X_test)[:, 0]
    filas = []

    for u in umbrales:
        # Si la probabilidad de impago supera el umbral, se clasifica como 0
        y_pred = np.where(proba_impago >= u, 0, 1)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        filas.append({
            "umbral": round(u, 2),
            "precision": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=0),
            "f1": f1_score(y_test, y_pred, pos_label=0),
            "detectados": tn,      # impagos correctamente identificados
            "escapados": fp,       # impagos no detectados
            "falsas_alarmas": fn   # buenos clientes rechazados
        })

    return pd.DataFrame(filas).set_index("umbral").round(4)

def validar_modelos(X, y, n_folds=5):
    """Evalúa cada modelo con validación cruzada estratificada.

    Un solo split puede dar un resultado favorable o desfavorable por azar.
    Con k particiones se obtienen k mediciones, y su desviación indica si el
    desempeño es estable.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)

    # make_scorer permite fijar pos_label=0 en las métricas
    from sklearn.metrics import make_scorer
    scoring = {
        "f1": make_scorer(f1_score, pos_label=0),
        "recall": make_scorer(recall_score, pos_label=0),
        "precision": make_scorer(precision_score, pos_label=0, zero_division=0),
        "roc_auc": "roc_auc"
    }

    filas = []
    for nombre, modelo in construir_modelos(y).items():
        print(f"Validando {nombre}...")
        pipe = Pipeline([
            ("preprocesador", construir_preprocesador()),
            ("modelo", modelo)
        ])
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        filas.append({
            "modelo": nombre,
            "f1_media": res["test_f1"].mean(),
            "f1_std": res["test_f1"].std(),
            "recall_media": res["test_recall"].mean(),
            "roc_auc_media": res["test_roc_auc"].mean(),
            "roc_auc_std": res["test_roc_auc"].std()
        })

    return pd.DataFrame(filas).set_index("modelo").round(4)

def ajustar_random_forest(X_train, y_train):
    """Busca la combinación de hiperparámetros que maximiza F1 sobre la clase 0.

    El RandomForest sin restricciones alcanza recall de 0,029: con 409 impagos
    en entrenamiento, los árboles crecen hasta aislar casos individuales en
    lugar de aprender un patrón generalizable. Limitar la profundidad y exigir
    un mínimo de muestras por hoja fuerza reglas más generales.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    pipe = Pipeline([
        ("preprocesador", construir_preprocesador()),
        ("modelo", RandomForestClassifier(
            class_weight="balanced", random_state=SEMILLA, n_jobs=-1))
    ])

    # El prefijo 'modelo__' indica que el parámetro es del paso llamado 'modelo'
    grilla = {
        "modelo__n_estimators": [100, 200],
        "modelo__max_depth": [3, 5, 8],
        "modelo__min_samples_leaf": [20, 50]
    }

    busqueda = GridSearchCV(
        pipe, grilla,
        scoring=make_scorer(f1_score, pos_label=0),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA),
        n_jobs=-1,
        verbose=1
    )
    busqueda.fit(X_train, y_train)

    return busqueda

if __name__ == "__main__":
    X, y = preparar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)

    print(f"Train: {len(X_train)} registros ({(y_train == 0).sum()} impagos)")
    print(f"Test:  {len(X_test)} registros ({(y_test == 0).sum()} impagos)\n")

    # --- Comparación inicial ---
    tabla, pipelines = entrenar_y_evaluar(X_train, X_test, y_train, y_test)

    # --- Validación cruzada ---
    print("\n=== Validación cruzada (5 folds) ===")
    print(validar_modelos(X, y))

    # --- Ajuste de hiperparámetros ---
    print("\n=== Ajuste de hiperparámetros (RandomForest) ===")
    busqueda = ajustar_random_forest(X_train, y_train)
    print(f"Mejores parámetros: {busqueda.best_params_}")

    rf_ajustado = busqueda.best_estimator_
    y_pred_rf = rf_ajustado.predict(X_test)
    y_proba_rf = rf_ajustado.predict_proba(X_test)[:, 0]

    # Se incorpora a la comparación general
    metricas_rf = calcular_metricas(y_test, y_pred_rf, y_proba_rf)
    metricas_rf["modelo"] = "RandomForest_ajustado"
    tabla = pd.concat([
        tabla,
        pd.DataFrame([metricas_rf]).set_index("modelo")
    ]).round(4)
    pipelines["RandomForest_ajustado"] = rf_ajustado

    print("\n=== Comparación final ===")
    print(tabla)

    # --- Selección ---
    # El RandomForest ajustado alcanza mejor F1, pero LogisticRegression
    # detecta más impagos. En crédito el costo de un falso negativo —el
    # capital prestado— supera al de un falso positivo —la ganancia del
    # crédito no otorgado—, por lo que se prioriza recall sobre F1.
    # Los valores concretos figuran en la tabla impresa arriba.
    MODELO_FINAL = "LogisticRegression"
    pipe_mejor = pipelines[MODELO_FINAL]

    print(f"\nMejor F1:            {tabla['f1'].idxmax()}")
    print(f"Modelo seleccionado: {MODELO_FINAL} (por recall sobre la clase 0)")

    # --- Detalle del modelo elegido ---
    y_pred = pipe_mejor.predict(X_test)

    print("\n=== Matriz de confusión ===")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["Real: impago", "Real: pago"],
        columns=["Pred: impago", "Pred: pago"]
    ))

    print("\n=== Reporte completo ===")
    print(classification_report(y_test, y_pred, target_names=["impago", "pago"]))

    # --- Umbral ---
    print("\n=== Optimización del umbral de decisión ===")
    tabla_umbral = optimizar_umbral(pipe_mejor, X_test, y_test)
    print(tabla_umbral)
    print(f"\nUmbral con mejor F1: {tabla_umbral['f1'].idxmax()}")

    # --- Guardado ---
    ruta_modelo = Path(__file__).resolve().parents[2] / "modelo_final.pkl"
    joblib.dump(pipe_mejor, ruta_modelo)
    print(f"\nModelo guardado en {ruta_modelo.name}")