# Modelo de riesgo crediticio

Proyecto Integrador del Módulo 5 (MLOps y Deployment) — Soy Henry.

Modelo de clasificación que estima la probabilidad de que una solicitud de
crédito no sea pagada a tiempo, desplegado como pipeline reproducible.

---

## El problema

Una financiera necesita anticipar qué solicitantes incumplirán el pago de su
crédito. El dataset contiene **10.763 créditos históricos** con 23 variables
cada uno.

La variable objetivo es `Pago_atiempo`: 1 si el crédito se pagó a tiempo, 0 si
no. **El 95,25% pagó** — solo 511 registros corresponden a impago.

Ese desbalance condiciona todo el proyecto: un modelo que prediga "paga" para
todos alcanza 95,25% de accuracy sin detectar un solo cliente riesgoso. Por eso
la evaluación se apoya en precision, recall, F1 y ROC-AUC sobre la clase
minoritaria, y nunca en accuracy.

---

## Hallazgos principales

### Fuga de información en `puntaje`

El 87,4% de los registros comparte el valor `95.227787`. Al cruzarla con la
variable objetivo, la separación resulta perfecta: el máximo de `puntaje` entre
los impagos es **62,67** y el mínimo entre quienes pagaron es **63,81**. Un
umbral en 63 clasifica correctamente el 100% de los registros.

Ninguna variable legítima predice sin error. La separación indica que el valor
deriva del resultado o se calculó con posterioridad. **Se excluyó del modelo**:
mantenerla habría producido métricas cercanas a la perfección sin capacidad
real de generalización.

### Fuga en las variables de mora

`saldo_mora` registra el monto actualmente vencido — información posterior al
incumplimiento, no disponible al evaluar una solicitud nueva. Sus 3 casos de
mora de codeudor son además un subconjunto exacto de los 55 del titular.
**Ambas se excluyeron.**

### El cero como dato faltante

El dataset utiliza 0 para representar ausencia de dato en varias columnas: 24
salarios en cero, 145 puntajes de Datacrédito en cero. Se unificaron como
valores faltantes.

### Los faltantes son informativos

2.930 registros carecen de datos de Datacrédito, y la ausencia no es aleatoria:
afecta al 41% de los independientes frente al 19% de los empleados. Esos
registros presentan **5,73% de impago contra 4,38%** (chi-cuadrado,
p = 0,0038), por lo que se incorporó un indicador binario que preserva la
condición al imputar.

### Ninguna variable separa las clases

Cinco variables muestran diferencia estadísticamente significativa
(Mann-Whitney): `puntaje_datacredito`, `huella_consulta`, `edad_cliente`,
`promedio_ingresos_datacredito` y `capital_prestado`. Pero sus distribuciones
se superponen: el impago no se explica por ninguna variable aislada.

Es el comportamiento esperable tras eliminar las fugas.

---

## Resultados

| Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **LogisticRegression** | 0,076 | **0,569** | 0,134 | **0,691** |
| RandomForest | 1,000 | 0,039 | 0,076 | 0,661 |
| XGBoost | 1,000 | 0,029 | 0,057 | 0,656 |
| RandomForest ajustado | 0,099 | 0,373 | **0,156** | 0,681 |

Métricas sobre la clase 0 (impago), en el conjunto de prueba.

Validación cruzada de 5 particiones: F1 de **0,137 ± 0,007** y ROC-AUC de
**0,668 ± 0,017** para LogisticRegression. Los resultados son estables.

### Modelo seleccionado: LogisticRegression

El RandomForest ajustado alcanza mejor F1, pero detecta **38 impagos frente a
58**. En crédito, el costo de un falso negativo —el capital prestado— supera al
de un falso positivo —la ganancia de un crédito no otorgado—, por lo que se
priorizó recall sobre F1.

Matriz de confusión del modelo elegido:

|  | Pred: impago | Pred: pago |
|---|---|---|
| **Real: impago** | 58 | 44 |
| **Real: pago** | 704 | 1.347 |

### El umbral es ajustable

| Umbral | Detectados | Escapados | Falsas alarmas |
|---|---|---|---|
| 0,40 | 88 | 14 | 1.180 |
| 0,45 | 74 | 28 | 931 |
| **0,50** (actual) | **58** | **44** | **704** |
| 0,60 | 38 | 64 | 326 |

El máximo F1 se alcanza en 0,60, pero bajar el umbral detecta más impagos. La
elección final depende del costo relativo asignado a cada tipo de error.

---

## Monitoreo de drift

Comparación de distribuciones mediante Kolmogorov-Smirnov (numéricas) y
chi-cuadrado (categóricas), con dos particiones:

| Partición | Variables con drift |
|---|---|
| Aleatoria (control) | 1 de 25 |
| Cronológica | 4 de 25 |

El control valida el método: con 25 pruebas, 1 positivo es atribuible al azar.
El contraste confirma que el drift temporal detectado en la partición
cronológica es real.

El criterio de drift combina Kolmogorov-Smirnov (KS > 0,25) y Population
Stability Index (PSI > 0,25), en lugar de basarse en el p-valor: con 5.381
registros por grupo, el p-valor detecta diferencias mínimas sin magnitud real
—**significativo no equivale a relevante**—, mientras que KS y PSI miden el
tamaño del cambio.

Las dos variables con drift son `plazo_meses` (KS 0,218, PSI 0,348) y
`promedio_ingresos_datacredito` (KS 0,185, PSI 0,387). En `plazo_meses`, PSI
detecta un cambio que KS por poco no alcanza a marcar, lo que justifica usar
ambas métricas en conjunto.

El dashboard de Streamlit incluye un indicador de nivel de alerta (sin drift,
moderado, crítico) y un gráfico que compara la distribución histórica contra
la actual para la variable que se elija. Cada corrida de `model_monitoring.py`
persiste su resultado en `reports/drift_report.json` y acumula un historial en
`reports/drift_history.csv`, para poder auditar la evolución sin depender de
que el dashboard esté abierto.

---

## Estructura

```
mlops_pipeline/src/
├── Cargar_datos.ipynb              ingesta del Excel y exportación a CSV
├── comprension_eda.ipynb           exploración, limpieza y EDA
├── ft_engineering.py               pipeline de preparación de datos
├── model_training_evaluation.py    entrenamiento y selección
├── model_monitoring.py             detección de drift
├── model_deploy.py                 API (avance 4)
└── app_streamlit.py                interfaz de usuario
```

---

## Ejecución

```bash
# Entorno
py -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Pipeline
cd mlops_pipeline/src
python ft_engineering.py              # verifica el preprocesamiento
python model_training_evaluation.py   # entrena y genera modelo_final.pkl
python model_monitoring.py            # análisis de drift

# Interfaz
streamlit run app_streamlit.py
```

`modelo_final.pkl` no se versiona (está en `.gitignore`). Se genera al ejecutar
el entrenamiento, que es requisito previo para la aplicación.

### API de inferencia

```bash
cd mlops_pipeline/src
uvicorn model_deploy:app --reload
```

Documentación interactiva en http://localhost:8000/docs

| Endpoint          | Método | Descripción                       |
|-------------------|--------|------------------------------------|
| `/`               | GET    | Información del servicio          |
| `/health`         | GET    | Verificación de disponibilidad    |
| `/predecir`       | POST   | Evaluación de una solicitud       |
| `/predecir_batch` | POST   | Evaluación de varias solicitudes a la vez |

La validación de entrada se realiza con Pydantic sobre las mismas reglas
definidas en el EDA: edad entre 18 y 90, salario superior a 100.000, puntaje
entre 150 y 950. Una solicitud fuera de rango se rechaza con código 422 sin
llegar al modelo.

### Contenedor

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

La imagen parte de `python:3.12-slim` e instala las dependencias fijadas en
`requirements.txt`, de modo que el servicio se ejecuta igual en cualquier
máquina con Docker, con independencia de la versión de Python instalada
localmente.

`modelo_final.pkl` debe existir antes de construir la imagen: se genera con
`model_training_evaluation.py`.
---

## Decisiones y limitaciones

**Sin diccionario de datos.** La interpretación de `tipo_credito`,
`huella_consulta` o `puntaje` se apoya en convenciones del dominio crediticio.
Condiciona la traducción de hallazgos a decisiones de negocio, no el modelado.

**`tendencia_ingresos` contiene 58 valores numéricos** donde deberían figurar
las tres categorías. Se agruparon en una categoría `valor_anomalo` en lugar de
recodificarlos por signo: la interpretación de esos valores no es verificable
con los datos disponibles.

**Descenso de volumen sin explicación.** El dataset pasa de 1.917 créditos en
enero de 2025 a 11 en abril de 2026. Se evaluaron dos hipótesis —maduración de
la cartera y corte en la extracción— y ninguna es compatible con los datos. Se
documenta el patrón sin atribuirle causa.

**Los atributos derivados se calculan en dos lugares:** en `ft_engineering.py`
para el entrenamiento y en `app_streamlit.py` para las solicitudes
individuales. Es un riesgo de *feature drift* que convendría resolver
centralizando el cálculo.

**Herramientas aplicadas fuera del temario del módulo:** pruebas de
chi-cuadrado, binomial y Mann-Whitney para validar diferencias observadas, y
transformación logarítmica sobre las variables de monto (reduce el sesgo de
`capital_prestado` de 3,72 a 0,11).
