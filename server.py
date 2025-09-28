from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from validator import TrafficModelValidator

# Crear app FastAPI
app = FastAPI(
    title="Traffic Prediction API",
    description="API para validar y predecir congestión vial usando XGBoost",
    version="1.0"
)

# Instanciar el validador global
validator = TrafficModelValidator()

# ✅ Modelo de entrada para validar parquet
class ValidationRequest(BaseModel):
    parquet_path: str
    n_samples: int = 100

# ✅ Modelo de entrada para predicción (lo que viene desde tu formulario web)
class PredictionRequest(BaseModel):
    severity: int
    start_lat: float
    start_lng: float
    distance_mi: float
    delay_from_typical_traffic_mins: float
    delay_from_free_flow_speed_mins: float
    county: str
    state: str
    local_time_zone: str
    temperature_f: float = 70.0
    windchill_f: float = 70.0
    humidity: float = 50.0
    pressure_in: float = 29.9
    visibility_mi: float = 10.0
    wind_dir: str = "Calm"
    windspeed_mph: float = 5.0
    precipitation_in: float = 0.0
    weather_conditions: str = "Clear"
    hour: int = 8
    day_of_week: int = 2
    month: int = 5
    year: int = 2018
    is_weekend: int = 0
    duration_mins: float = 60.0

# 🩺 Health check
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": validator.xgb_model is not None}

# 🚀 Validación usando parquet (ya lo tienes)
@app.post("/validate/path")
def validate_from_path(req: ValidationRequest):
    if not os.path.exists(req.parquet_path):
        raise HTTPException(status_code=400, detail="Ruta del parquet no existe")
    try:
        success = validator.validate_model(req.parquet_path, req.n_samples)
        if not success:
            raise HTTPException(status_code=400, detail="Validación fallida. Revisa logs en consola.")
        return {"message": "✅ Validación completada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en validación: {str(e)}")

# 🔮 Nuevo endpoint: predicción individual desde formulario
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # 1️⃣ Crear DataFrame con una sola fila desde el JSON recibido
        df = pd.DataFrame([{
            "Severity": req.severity,
            "Start_Lat": req.start_lat,
            "Start_Lng": req.start_lng,
            "Distance(mi)": req.distance_mi,
            "DelayFromTypicalTraffic(mins)": req.delay_from_typical_traffic_mins,
            "DelayFromFreeFlowSpeed(mins)": req.delay_from_free_flow_speed_mins,
            "County": validator.county_map.get(req.county, validator.county_map.get("Other", 0)),
            "State": validator.state_map.get(req.state, 0),
            "LocalTimeZone": validator.timezone_map.get(req.local_time_zone, 0),
            "Temperature(F)": req.temperature_f,
            "WindChill(F)": req.windchill_f,
            "Humidity(%)": req.humidity,
            "Pressure(in)": req.pressure_in,
            "Visibility(mi)": req.visibility_mi,
            "WindDir": validator.winddir_map.get(req.wind_dir, 0),
            "WindSpeed(mph)": req.windspeed_mph,
            "Precipitation(in)": req.precipitation_in,
            "Weather_Conditions": validator.weather_map.get(req.weather_conditions, validator.weather_map.get("Other", 0)),
            "Hour": req.hour,
            "DayOfWeek": req.day_of_week,
            "Month": req.month,
            "Year": req.year,
            "IsWeekend": req.is_weekend,
            "Duration(mins)": req.duration_mins
        }])

        # 2️⃣ Predecir con el modelo cargado
        y_pred = validator.xgb_model.predict(df)[0]
        y_proba = validator.xgb_model.predict_proba(df)[0]

        # 3️⃣ Decodificar clase
        predicted_class = validator.target_code_to_text[y_pred]

        return {
            "status": "success",
            "prediction": {
                "predicted_class": predicted_class,
                "confidence": float(y_proba[y_pred]),
                "class_probabilities": {
                    cls: float(prob) for cls, prob in zip(validator.target_code_to_text.values(), y_proba)
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# 📊 Comparación detallada: Real vs Predicho
@app.post("/validate/compare")
def compare_predictions(req: ValidationRequest):
    if not os.path.exists(req.parquet_path):
        raise HTTPException(status_code=400, detail="Ruta del parquet no existe")

    try:
        # 1️⃣ Cargar parquet original
        df_raw = pd.read_parquet(req.parquet_path, engine="pyarrow")

        if "Congestion_Speed" not in df_raw.columns:
            raise HTTPException(status_code=400, detail="El parquet no contiene la columna 'Congestion_Speed'")

        # 2️⃣ Tomar muestra
        df_sample = df_raw.sample(n=req.n_samples, random_state=42)

        # 3️⃣ Guardar target real
        y_real = df_sample["Congestion_Speed"].copy()

        # 4️⃣ Procesar datos igual que en el validador
        X_processed, _ = validator.process_data_for_model(df_sample)

        # 5️⃣ Predecir
        y_pred = validator.xgb_model.predict(X_processed)
        y_proba = validator.xgb_model.predict_proba(X_processed)

        # 6️⃣ Mapear códigos → texto
        y_pred_text = [validator.target_code_to_text[i] for i in y_pred]

        # 7️⃣ Construir dataframe de resultados
        resultados = pd.DataFrame({
            "Real_Congestion_Speed": y_real.values,
            "Predicted_Congestion_Speed": y_pred_text,
            "Is_Correct": y_real.values == y_pred_text,
            "Confidence": y_proba.max(axis=1),
        })

        # Agregar probabilidades por clase
        for idx, cls_name in validator.target_code_to_text.items():
            resultados[f"Prob_{cls_name}"] = y_proba[:, idx]

        # Guardar CSV para inspección si quieres
        resultados.to_csv("comparison_results.csv", index=False)

        # Resumen rápido
        accuracy = (resultados["Real_Congestion_Speed"] == resultados["Predicted_Congestion_Speed"]).mean()

        return {
            "status": "success",
            "total_samples": len(resultados),
            "accuracy": round(float(accuracy) * 100, 2),
            "preview": resultados.head(10).to_dict(orient="records"),
            "csv_path": "comparison_results.csv"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la comparación: {str(e)}")
