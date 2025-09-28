import pandas as pd
import requests
import numpy as np

PARQUET_PATH = r"C:/2025/CURSOS UNMSM/CICLO 8/SI/Tarea - Modelo ML/dataset_brotli.parquet"
API_URL = "http://127.0.0.1:8000/predict"

# Leer el parquet original
df = pd.read_parquet(PARQUET_PATH)
df = df.sample(100, random_state=42)  # 100 registros al azar
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

resultados = []

for idx, row in df.iterrows():
    try:
        # Construimos el payload con todas las features necesarias
        payload = {
            "severity": int(row["Severity"]),
            "start_lat": float(row.get("Start_Lat", 0)),
            "start_lng": float(row.get("Start_Lng", 0)),
            "distance_mi": float(row["Distance(mi)"]),
            "delay_from_typical_traffic_mins": float(row.get("DelayFromTypicalTraffic(mins)", 0)),
            "delay_from_free_flow_speed_mins": float(row.get("DelayFromFreeFlowSpeed(mins)", 0)),
            "county": str(row["County"]),
            "state": str(row["State"]),
            "local_time_zone": str(row.get("LocalTimeZone", "US/Eastern")),
            "temperature_f": float(row.get("Temperature(F)", 70.0)),
            "windchill_f": float(row.get("WindChill(F)", 70.0)),
            "humidity": float(row.get("Humidity(%)", 50.0)),
            "pressure_in": float(row.get("Pressure(in)", 29.9)),
            "visibility_mi": float(row.get("Visibility(mi)", 10.0)),
            "wind_dir": str(row.get("WindDir", "Calm")),
            "windspeed_mph": float(row.get("WindSpeed(mph)", 5.0)),
            "precipitation_in": float(row.get("Precipitation(in)", 0.0)),
            "weather_conditions": str(row.get("Weather_Conditions", "Clear")),
            "hour": int(row.get("Hour", 8)),
            "day_of_week": int(row.get("DayOfWeek", 2)),
            "month": int(row.get("Month", 5)),
            "year": int(row.get("Year", 2018)),
            "is_weekend": int(row.get("IsWeekend", 0)),
            "duration_mins": float(row.get("Duration(mins)", 60.0)),
        }

        # Llamada al endpoint de predicción
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        pred_data = response.json()["prediction"]

        resultados.append({
            **payload,  # todas las features que entraron al modelo
            "Real_Congestion_Speed": row["Congestion_Speed"],  # target real
            "Predicted_Congestion_Speed": pred_data["predicted_class"],  # predicho
            "Confidence": pred_data["confidence"],  # confianza de la clase predicha
        })

    except Exception as e:
        print(f"❌ Error en fila {idx}: {e}")

# Crear el DataFrame con todos los datos
df_resultados = pd.DataFrame(resultados)

# Mostrar primeras filas
print("\n📊 Comparación detallada (primeras 10 filas):")
print(df_resultados.head(10))

# Guardar CSV completo
df_resultados.to_csv("comparacion_detallada.csv", index=False, encoding="utf-8-sig")
print("✅ Resultados guardados en 'comparacion_detallada.csv'")
