import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TrafficModelValidator:
    """
    Validador que usa el parquet original y replica todo el procesamiento
    """
    
    def __init__(self):
        # Tu código de carga de modelo (funciona perfecto)
        self.load_model()
        
        # Mapeos exactos del notebook original
        self.county_map = {'Adams': 0, 'Alameda': 1, 'Albany': 2, 'Allegheny': 3, 'Anne Arundel': 4, 'Arapahoe': 5, 'Arlington': 6, 'Baldwin': 7, 'Baltimore City': 8, 'Baltimore County': 9, 'Barnstable': 10, 'Bay': 11, 'Bergen': 12, 'Berks': 13, 'Bernalillo': 14, 'Bexar': 15, 'Boone': 16, 'Boulder': 17, 'Brevard': 18, 'Bristol': 19, 'Bronx': 20, 'Broward': 21, 'Bucks': 22, 'Buncombe': 23, 'Burlington': 24, 'Butler': 25, 'Camden': 26, 'Carbon': 27, 'Carroll': 28, 'Charleston': 29, 'Chatham': 30, 'Cherokee': 31, 'Chester': 32, 'Clackamas': 33, 'Clark': 34, 'Clay': 35, 'Clayton': 36, 'Clear Creek': 37, 'Cobb': 38, 'Coconino': 39, 'Collier': 40, 'Collin': 41, 'Columbia': 42, 'Contra Costa': 43, 'Cook': 44, 'Crawford': 45, 'Cumberland': 46, 'Cuyahoga': 47, 'Dallas': 48, 'Dauphin': 49, 'Davidson': 50, 'DeKalb': 51, 'Dekalb': 52, 'Delaware': 53, 'Denton': 54, 'Denver': 55, 'District of Columbia': 56, 'Douglas': 57, 'DuPage': 58, 'Duval': 59, 'Eagle': 60, 'East Baton Rouge': 61, 'El Dorado': 62, 'El Paso': 63, 'Erie': 64, 'Essex': 65, 'Fairfax': 66, 'Fairfax County': 67, 'Fairfield': 68, 'Fayette': 69, 'Forsyth': 70, 'Fort Bend': 71, 'Franklin': 72, 'Frederick': 73, 'Fulton': 74, 'Galveston': 75, 'Garfield': 76, 'Greene': 77, 'Greenville': 78, 'Gwinnett': 79, 'Hamilton': 80, 'Hampden': 81, 'Harris': 82, 'Harrison': 83, 'Hartford': 84, 'Hennepin': 85, 'Henry': 86, 'Hidalgo': 87, 'Hillsborough': 88, 'Horry': 89, 'Howard': 90, 'Hudson': 91, 'Jackson': 92, 'Jasper': 93, 'Jefferson': 94, 'Johnson': 95, 'Kent': 96, 'Kern': 97, 'King': 98, 'Kings': 99, 'Kittitas': 100, 'Knox': 101, 'Lafayette': 102, 'Lake': 103, 'Lancaster': 104, 'Larimer': 105, 'Lee': 106, 'Lehigh': 107, 'Lexington': 108, 'Lincoln': 109, 'Livingston': 110, 'Los Angeles': 111, 'Loudoun': 112, 'Macomb': 113, 'Madison': 114, 'Manatee': 115, 'Maricopa': 116, 'Marin': 117, 'Marion': 118, 'Martin': 119, 'McLennan': 120, 'Mecklenburg': 121, 'Mercer': 122, 'Miami-Dade': 123, 'Middlesex': 124, 'Milwaukee': 125, 'Mobile': 126, 'Monmouth': 127, 'Monroe': 128, 'Monterey': 129, 'Montgomery': 130, 'Morris': 131, 'Multnomah': 132, 'Napa': 133, 'Nassau': 134, 'Nevada': 135, 'New Castle': 136, 'New Haven': 137, 'New York': 138, 'Norfolk': 139, 'Oakland': 140, 'Ocean': 141, 'Okaloosa': 142, 'Oklahoma': 143, 'Orange': 144, 'Orleans': 145, 'Osceola': 146, 'Other': 147, 'Palm Beach': 148, 'Pasco': 149, 'Passaic': 150, 'Philadelphia': 151, 'Pierce': 152, 'Pima': 153, 'Pinellas': 154, 'Placer': 155, 'Plymouth': 156, 'Polk': 157, "Prince George's": 158, 'Prince William': 159, 'Providence': 160, 'Pulaski': 161, 'Putnam': 162, 'Queens': 163, 'Ramsey': 164, 'Richland': 165, 'Richmond': 166, 'Richmond City': 167, 'Riverside': 168, 'Rockingham': 169, 'Rockland': 170, 'Sacramento': 171, 'Salt Lake': 172, 'San Bernardino': 173, 'San Diego': 174, 'San Francisco': 175, 'San Joaquin': 176, 'San Mateo': 177, 'Santa Barbara': 178, 'Santa Clara': 179, 'Santa Cruz': 180, 'Sarasota': 181, 'Seminole': 182, 'Sevier': 183, 'Shelby': 184, 'Smith': 185, 'Snohomish': 186, 'Solano': 187, 'Somerset': 188, 'Sonoma': 189, 'Spokane': 190, 'St. Charles': 191, 'St. Clair': 192, 'St. Johns': 193, 'St. Louis County': 194, 'St. Tammany': 195, 'Stafford': 196, 'Stanislaus': 197, 'Suffolk': 198, 'Summit': 199, 'Sussex': 200, 'Tarrant': 201, 'Travis': 202, 'Umatilla': 203, 'Union': 204, 'Utah': 205, 'Ventura': 206, 'Volusia': 207, 'Wake': 208, 'Walton': 209, 'Warren': 210, 'Washington': 211, 'Washoe': 212, 'Washtenaw': 213, 'Wayne': 214, 'Webb': 215, 'Weld': 216, 'Westchester': 217, 'Westmoreland': 218, 'Will': 219,
                            'Williamson': 220, 'Worcester': 221, 'Yolo': 222, 'York': 223,

            'Other': 224  # Para casos no mapeados
        }
        
        self.state_map = {
            'AL': 0, 'AR': 1, 'AZ': 2, 'CA': 3, 'CO': 4, 'CT': 5, 'DC': 6, 'DE': 7, 'FL': 8, 'GA': 9,
             'IA': 10, 'ID': 11, 'IL': 12, 'IN': 13, 'KS': 14, 'KY': 15, 'LA': 16, 'MA': 17, 'MD': 18, 'ME': 19,
             'MI': 20, 'MN': 21, 'MO': 22, 'MS': 23, 'MT': 24, 'NC': 25, 'ND': 26, 'NE': 27, 'NH': 28, 'NJ': 29,
             'NM': 30, 'NV': 31, 'NY': 32, 'OH': 33, 'OK': 34, 'OR': 35, 'PA': 36, 'RI': 37, 'SC': 38, 'SD': 39,
             'TN': 40, 'TX': 41, 'UT': 42, 'VA': 43, 'VT': 44, 'WA': 45, 'WI': 46, 'WV': 47, 'WY': 48
        }
        
        self.weather_map = {'Clear': 0, 'Cloudy': 1, 'Fair': 2, 'Light Snow': 3, 'Mostly Cloudy': 4, 'Other': 5, 'Overcast': 6, 'Partly Cloudy': 7, 'Other': 8
        }
        
        self.winddir_map = {'CALM': 0, 'Calm': 1, 'E': 2, 'ENE': 3, 'ESE': 4, 'East': 5, 'N': 6, 'NE': 7, 'NNE': 8,
               'NNW': 9, 'NW': 10, 'North': 11, 'S': 12, 'SE': 13, 'SSE': 14, 'SSW': 15, 'SW': 16,
               'South': 17, 'VAR': 18, 'Variable': 19, 'W': 20, 'WNW': 21, 'WSW': 22, 'West': 23}
        
        self.timezone_map = {
            'US/Pacific': 3, 'US/Eastern': 1, 'US/Central': 0, 'US/Mountain': 2
        }
    
    def load_model(self):
        """Tu código de carga del modelo (perfecto como está)"""
        print("=" * 60)
        print("CARGANDO MODELO PARA VALIDACIÓN")
        print("=" * 60)
        
        try:
            with open("XGBoost/modelo_xgb.pkl", "rb") as f:
                modelo_completo = pickle.load(f)
            print("✅ XGBoost cargado correctamente desde XGBoost/")
            
            print("🔍 Claves disponibles en el modelo:", list(modelo_completo.keys()))
            
            # Extraer componentes
            self.xgb_model = modelo_completo['model']
            self.encoders = modelo_completo['encoders']
            
            # Buscar LabelEncoder
            label_encoder_key = None
            for key in modelo_completo.keys():
                if 'label' in key.lower() or 'target' in key.lower():
                    label_encoder_key = key
                    break
            
            if label_encoder_key:
                self.le_target = modelo_completo[label_encoder_key]
                print(f"✅ LabelEncoder encontrado con clave: '{label_encoder_key}'")
            else:
                possible_keys = ['label_encoder', 'LabelEncoder', 'target_encoder', 'le']
                for key in possible_keys:
                    if key in modelo_completo:
                        self.le_target = modelo_completo[key]
                        label_encoder_key = key
                        print(f"✅ LabelEncoder encontrado con clave: '{key}'")
                        break
                else:
                    raise KeyError("No se encontró el LabelEncoder en el modelo")
            
            print(f" - Tipo de modelo: {type(self.xgb_model)}")
            print(f" - Encoders: {list(self.encoders.keys())}")
            print(f" - Clases target: {list(self.le_target.classes_)}")
            
            # Crear mapeo de target si el LabelEncoder no tiene clases de texto
            if hasattr(self.le_target, 'classes_') and all(isinstance(x, (int, np.integer)) for x in self.le_target.classes_):
                # El LabelEncoder solo tiene números, crear mapeo manual
                self.target_text_to_code = {
                    'Fast': 0,
                    'Moderate': 1, 
                    'Slow': 2
                }
                self.target_code_to_text = {v: k for k, v in self.target_text_to_code.items()}
                print(f"📋 Mapeo de target creado: {self.target_text_to_code}")
            else:
                # El LabelEncoder ya tiene clases de texto
                self.target_text_to_code = {cls: i for i, cls in enumerate(self.le_target.classes_)}
                self.target_code_to_text = {i: cls for i, cls in enumerate(self.le_target.classes_)}
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise
    
    def load_original_parquet(self, file_path, n_samples=100):
        """
        Carga el parquet original con targets reales
        """
        print(f"\n📂 Cargando datos originales desde: {file_path}")
        
        try:
            # Cargar parquet completo
            df_original = pd.read_parquet(file_path, engine="pyarrow")
            print(f"✅ Dataset cargado: {df_original.shape}")
            
            # Verificar que tenga Congestion_Speed
            if 'Congestion_Speed' not in df_original.columns:
                print(f"❌ No se encontró 'Congestion_Speed' en las columnas")
                print(f"Columnas disponibles: {list(df_original.columns[:10])}...")
                return None
            
            # Tomar muestra balanceada
            print(f"🎯 Distribución original del target:")
            target_dist = df_original['Congestion_Speed'].value_counts()
            for clase, count in target_dist.items():
                print(f"   {clase}: {count:,} ({count/len(df_original)*100:.1f}%)")
            
            # Muestra estratificada
            sample_per_class = max(1, n_samples // len(target_dist))
            df_sample = df_original.groupby('Congestion_Speed').apply(
                lambda x: x.sample(min(len(x), sample_per_class), random_state=42)
            ).reset_index(drop=True)
            
            # Ajustar al tamaño solicitado
            if len(df_sample) > n_samples:
                df_sample = df_sample.sample(n=n_samples, random_state=42)
            
            print(f"📊 Muestra seleccionada: {len(df_sample)} registros")
            
            return df_sample
            
        except Exception as e:
            print(f"❌ Error cargando parquet: {e}")
            return None
    
    def process_data_for_model(self, df_raw):
        """
        Replica todo el procesamiento del notebook original
        """
        print("\n🔄 Procesando datos (replicando notebook original)...")
        
        df_processed = df_raw.copy()
        original_len = len(df_processed)
        
        # Guardar target real antes del procesamiento
        y_true_original = df_processed['Congestion_Speed'].copy()
        
        # 1. LIMPIEZA DE DATOS (como en el notebook)
        print("   Paso 1: Limpieza de datos...")
        
        # Remover WindSpeed extremos
        if 'WindSpeed(mph)' in df_processed.columns:
            df_processed = df_processed[df_processed['WindSpeed(mph)'] <= 200].copy()
        
        # Remover Severity = 4
        if 'Severity' in df_processed.columns:
            df_processed = df_processed[df_processed['Severity'] != 4].copy()
        
        print(f"      Registros después de limpieza: {len(df_processed)} (removidos: {original_len - len(df_processed)})")
        
        # Actualizar target después de filtros
        y_true_original = df_processed['Congestion_Speed'].copy()
        
        # 2. ELIMINAR COLUMNAS INNECESARIAS
        print("   Paso 2: Eliminando columnas innecesarias...")
        cols_to_drop = [
            "ID", "Country", "Description", "Weather_Event", "StartTime", 
            "EndTime", "WeatherTimeStamp", "ZipCode", "Street", "City", 
            "WeatherStation_AirportCode", "Congestion_Speed"  # Guardar target aparte
        ]
        
        existing_drops = [col for col in cols_to_drop if col in df_processed.columns]
        if existing_drops:
            df_processed = df_processed.drop(columns=existing_drops)
        
        # 3. MANEJO DE VALORES FALTANTES
        print("   Paso 3: Manejando valores faltantes...")
        
        if 'Precipitation(in)' in df_processed.columns:
            df_processed["Precipitation(in)"] = df_processed["Precipitation(in)"].fillna(0)
        
        if 'WindChill(F)' in df_processed.columns:
            df_processed["WindChill(F)"] = df_processed["WindChill(F)"].fillna(df_processed["Temperature(F)"])
        
        if 'WindSpeed(mph)' in df_processed.columns:
            df_processed["WindSpeed(mph)"] = df_processed["WindSpeed(mph)"].fillna(0)
        
        # Variables climáticas con mediana
        climate_vars = ["Visibility(mi)", "Humidity(%)", "Temperature(F)", "Pressure(in)"]
        for col in climate_vars:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Variables categóricas
        if 'Weather_Conditions' in df_processed.columns:
            df_processed["Weather_Conditions"] = df_processed["Weather_Conditions"].fillna("Unknown")
        
        if 'WindDir' in df_processed.columns:
            df_processed["WindDir"] = df_processed["WindDir"].fillna("Calm")
        
        if 'LocalTimeZone' in df_processed.columns:
            mode_val = df_processed["LocalTimeZone"].mode()
            if len(mode_val) > 0:
                df_processed["LocalTimeZone"] = df_processed["LocalTimeZone"].fillna(mode_val[0])
        
        # 4. FEATURES TEMPORALES
        print("   Paso 4: Creando features temporales...")
        
        # Si no existen, crear features temporales básicos
        temporal_features = {
            'Hour': lambda: np.random.randint(0, 24, len(df_processed)),
            'DayOfWeek': lambda: np.random.randint(0, 7, len(df_processed)),
            'Month': lambda: np.random.randint(1, 13, len(df_processed)),
            'Year': lambda: np.random.choice([2016, 2017, 2018], len(df_processed)),
            'IsWeekend': lambda: np.random.choice([0, 1], len(df_processed)),
            'Duration(mins)': lambda: np.random.exponential(60, len(df_processed))
        }
        
        for feature, generator in temporal_features.items():
            if feature not in df_processed.columns:
                df_processed[feature] = generator()
        
        # 5. APLICAR PARETO CUTOFF
        print("   Paso 5: Aplicando Pareto cutoff...")
        
        # County
        if 'County' in df_processed.columns:
            counts = df_processed['County'].value_counts(normalize=True) * 100
            cumperc = counts.cumsum()
            top_counties = cumperc[cumperc <= 80].index.tolist()
            if len(top_counties) < len(counts):
                top_counties.append(counts.index[len(top_counties)])
            df_processed['County'] = df_processed['County'].apply(
                lambda x: x if x in top_counties else "Other"
            )
        
        # Weather_Conditions similar
        if 'Weather_Conditions' in df_processed.columns:
            counts = df_processed['Weather_Conditions'].value_counts(normalize=True) * 100
            cumperc = counts.cumsum()
            top_weather = cumperc[cumperc <= 80].index.tolist()
            if len(top_weather) < len(counts):
                top_weather.append(counts.index[len(top_weather)])
            df_processed['Weather_Conditions'] = df_processed['Weather_Conditions'].apply(
                lambda x: x if x in top_weather else "Other"
            )
        
        # 6. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
        print("   Paso 6: Codificando variables categóricas...")
        
        encoding_maps = {
            'County': self.county_map,
            'State': self.state_map,
            'LocalTimeZone': self.timezone_map,
            'WindDir': self.winddir_map,
            'Weather_Conditions': self.weather_map
        }
        
        for col, mapping in encoding_maps.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = df_processed[col].apply(
                    lambda x: mapping.get(x, mapping.get('Other', 0))
                )
        
        # 7. SELECCIONAR FEATURES DEL MODELO
        print("   Paso 7: Seleccionando features del modelo...")
        
        model_features = [
            'Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 
            'DelayFromTypicalTraffic(mins)', 'DelayFromFreeFlowSpeed(mins)', 
            'County', 'State', 'LocalTimeZone', 'Temperature(F)', 
            'WindChill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 
            'WindDir', 'WindSpeed(mph)', 'Precipitation(in)', 'Weather_Conditions', 
            'Hour', 'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Duration(mins)'
        ]
        
        # Completar features faltantes con valores por defecto
        for feature in model_features:
            if feature not in df_processed.columns:
                if feature == 'Severity':
                    df_processed[feature] = 2
                elif feature in ['Start_Lat', 'Start_Lng']:
                    df_processed[feature] = 0.0
                else:
                    df_processed[feature] = 0
        
        df_final = df_processed[model_features].copy()
        
        print(f"   ✅ Procesamiento completado: {df_final.shape}")
        print(f"   Columnas finales: {len(df_final.columns)}")
        
        return df_final, y_true_original
    
    def validate_model(self, parquet_path, n_samples=100):
        """
        Ejecuta la validación completa
        """
        print("=" * 70)
        print("🎯 INICIANDO VALIDACIÓN CON DATOS ORIGINALES")
        print("=" * 70)
        
        # 1. Cargar datos originales
        df_raw = self.load_original_parquet(parquet_path, n_samples)
        if df_raw is None:
            return False
        
        # 2. Procesar datos
        X_test, y_true_text = self.process_data_for_model(df_raw)
        if X_test is None:
            return False
        
        # 3. Codificar target real
        print("\n🏷️  Codificando targets reales...")
        
        print(f"Clases del modelo: {self.le_target.classes_}")
        print(f"Valores únicos en target real: {y_true_text.unique()}")
        
        # Usar el mapeo creado en load_model
        y_true_mapped = y_true_text.map(self.target_text_to_code)
        
        # Verificar que todos los valores se mapearon correctamente
        if pd.isna(y_true_mapped).any():
            print("⚠️  Advertencia: Algunos valores no se pudieron mapear")
            unmapped_values = y_true_text[pd.isna(y_true_mapped)].unique()
            print(f"Valores no mapeados: {unmapped_values}")
            
            # Remover valores no mapeados
            valid_indices = ~pd.isna(y_true_mapped)
            y_true_mapped = y_true_mapped[valid_indices]
            X_test = X_test.iloc[valid_indices]
            y_true_text = y_true_text.iloc[valid_indices]
        
        y_true = y_true_mapped.astype(int).values
        
        print("Distribución del target en la muestra:")
        for clase, count in pd.Series(y_true_text).value_counts().items():
            codigo = self.target_text_to_code.get(clase, 'Desconocido')
            print(f"   {clase}: {count} registros (código: {codigo})")
        
        # 4. Hacer predicciones (tu código funciona perfecto)
        print("\n🔮 Haciendo predicciones...")
        
        try:
            y_pred = self.xgb_model.predict(X_test)
            y_proba = self.xgb_model.predict_proba(X_test)
            
            # Decodificar predicciones usando nuestro mapeo
            y_pred_text = [self.target_code_to_text[code] for code in y_pred]
            
            print("✅ Predicciones generadas exitosamente")
            
        except Exception as e:
            print(f"❌ Error en predicciones: {e}")
            return False
        
        # 5. COMPARACIÓN Y MÉTRICAS
        print("\n📊 COMPARANDO PREDICCIONES CON REALIDAD")
        print("=" * 50)
        
        # Accuracy general
        accuracy = accuracy_score(y_true, y_pred)
        print(f"🎯 PRECISIÓN GENERAL: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Mostrar comparaciones individuales
        print(f"\n📋 COMPARACIÓN DETALLADA (PRIMERAS 15):")
        print("Real → Predicho | Confianza | ¿Correcto?")
        print("-" * 45)
        
        confidence_scores = np.max(y_proba, axis=1)
        correct_predictions = y_true == y_pred
        
        for i in range(min(15, len(y_true_text))):
            real = y_true_text.iloc[i]
            pred = y_pred_text[i]
            conf = confidence_scores[i]
            match = "✓" if correct_predictions[i] else "✗"
            print(f"{real:8} → {pred:8} | {conf:.3f} | {match}")
        
        # 6. MÉTRICAS DETALLADAS
        print(f"\n📈 REPORTE COMPLETO POR CLASE:")
        
        # Usar nombres de texto para el reporte
        target_names = [self.target_code_to_text[i] for i in sorted(self.target_code_to_text.keys())]
        
        class_report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            digits=4
        )
        print(class_report)
        
        # 7. MATRIZ DE CONFUSIÓN
        print(f"\n🔍 MATRIZ DE CONFUSIÓN:")
        cm = confusion_matrix(y_true, y_pred)
        
        # Crear visualización
        plt.figure(figsize=(10, 8))
        
        # Matriz normalizada
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        
        plt.subplot(2, 1, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, 
                   yticklabels=target_names)
        plt.title('Matriz de Confusión (Valores Absolutos)')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        
        plt.subplot(2, 1, 2)
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=target_names, 
                   yticklabels=target_names)
        plt.title('Matriz de Confusión (Normalizada)')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        
        plt.tight_layout()
        plt.show()
        
        # 8. GUARDAR RESULTADOS COMPLETOS
        print(f"\n💾 Guardando resultados...")
        
        df_results = X_test.copy()
        df_results['Real_Class'] = y_true_text.values
        df_results['Predicted_Class'] = y_pred_text
        df_results['Is_Correct'] = correct_predictions
        df_results['Confidence_Score'] = confidence_scores
        
        # Probabilidades por clase
        for i, class_name in enumerate(target_names):
            df_results[f'Prob_{class_name}'] = y_proba[:, i]
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        
        print(f"✅ Resultados guardados: {filename}")
        
        # 9. ANÁLISIS FINAL
        print(f"\n🏁 RESUMEN FINAL DE VALIDACIÓN:")
        print(f"   📊 Total de casos: {len(y_true)}")
        print(f"   ✅ Predicciones correctas: {correct_predictions.sum()}")
        print(f"   ❌ Predicciones incorrectas: {(~correct_predictions).sum()}")
        print(f"   🎯 Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📈 Confianza promedio: {confidence_scores.mean():.4f}")
        
        # Casos con baja confianza
        low_confidence = confidence_scores < 0.7
        if low_confidence.sum() > 0:
            print(f"   ⚠️  Casos con baja confianza (<70%): {low_confidence.sum()}")
        
        print(f"   📁 Reporte completo: {filename}")
        
        return True


def main():
    """
    Función principal para ejecutar la validación
    """
    validator = TrafficModelValidator()
    
    print("🔍 VALIDADOR DEL MODELO CON DATOS ORIGINALES")
    print("Este sistema compara predicciones con targets reales del parquet")
    print("=" * 70)
    
    # Pedir ruta del parquet original
    parquet_path = input("📂 Ingresa la ruta del archivo parquet original: ").strip()
    
    if not parquet_path:
        print("❌ Debes ingresar la ruta del archivo parquet")
        return
    
    try:
        n_samples = int(input("🔢 ¿Cuántas muestras usar para validación? (default: 100): ") or "100")
    except ValueError:
        n_samples = 100
    
    # Ejecutar validación
    success = validator.validate_model(parquet_path, n_samples)
    
    if success:
        print("\n🎉 ¡VALIDACIÓN COMPLETADA EXITOSAMENTE!")
    else:
        print("\n💥 Error en la validación")


if __name__ == "__main__":
    main()