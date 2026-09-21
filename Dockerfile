# Imagen base: Python 3.12 en su versión reducida.
# Se fija la versión para que el contenedor sea reproducible.
FROM python:3.12-slim

WORKDIR /app

# Las dependencias se copian e instalan antes que el código: Docker guarda
# en caché cada paso, y así un cambio en el código no obliga a reinstalar
# todas las librerías.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y datos necesarios en tiempo de ejecución
COPY mlops_pipeline/ ./mlops_pipeline/
COPY Base_de_datos.csv .
COPY modelo_final.pkl .

WORKDIR /app/mlops_pipeline/src

EXPOSE 8000

# --host 0.0.0.0 permite recibir conexiones desde fuera del contenedor.
# Con el valor por defecto (127.0.0.1) el servicio solo sería accesible
# desde adentro.
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
