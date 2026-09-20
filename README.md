
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