

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import List
import uvicorn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pydantic import BaseModel


warnings.filterwarnings('ignore')

app = FastAPI(title="Smart Solar Energy Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
connected_clients: List[WebSocket] = []


class AdviceRequest(BaseModel):
    hours: int = 24
    battery_level: float | None = None
    avg_consumption: float | None = None


class RealDataMLManager:
    def __init__(self):
        # Random Forest
        self.solar_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.rf_metrics = None
        
        # Data
        self.data_loaded = False
        self.df = None
        
        # LSTM
        self.lstm_model = None
        self.lstm_scaler = MinMaxScaler()
        self.lstm_trained = False
        self.sequence_length = 24
        self.lstm_metrics = None
        
    def load_csv_data(self, file_path):
        """Load and preprocess your CSV data"""
        try:
            print(f"Loading data from {file_path}...")
            self.df = pd.read_csv(file_path)
            
            print(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            print("Columns:", list(self.df.columns))
            
            self.df = self.preprocess_data(self.df)
            self.data_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def preprocess_data(self, df):
        """Clean and prepare your specific dataset"""
        df = df.copy()
        
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
            df = df.sort_values('Datetime')
        
        if 'wind-direction' in df.columns:
            df['wind-direction'] = pd.to_numeric(df['wind-direction'], errors='coerce')
        
        if 'Datetime' in df.columns:
            df['hour'] = df['Datetime'].dt.hour
            df['day_of_year'] = df['Datetime'].dt.dayofyear
            df['month'] = df['Datetime'].dt.month
            df['day_of_week'] = df['Datetime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        if 'solar_mw' in df.columns:
            df['solar_kw'] = df['solar_mw']
        
        df = df.dropna()
        
        print(f"Data preprocessing complete. Clean dataset: {len(df)} rows")
        print("Sample data:")
        print(df.head())
        
        return df
    
    def train_solar_model(self):
        """Train Random Forest model using your real solar data"""
        if not self.data_loaded:
            return {"error": "No data loaded"}
        
        feature_columns = ['hour', 'day_of_year', 'month', 'day_of_week', 'is_weekend',
                          'wind-speed', 'humidity', 'average-wind-speed-(period)',
                          'average-pressure-(period)', 'temperature']
        
        available_features = [col for col in feature_columns if col in self.df.columns]
        print(f"Using features: {available_features}")
        
        X = self.df[available_features]
        y = self.df['solar_kw']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest model on real data...")
        self.solar_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.solar_model.fit(X_train_scaled, y_train)
        
        y_pred = self.solar_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = dict(zip(available_features, self.solar_model.feature_importances_))
        
        self.is_trained = True
        
        metrics = {
            'model_type': 'Random Forest',
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': available_features
        }
        
        self.rf_metrics = metrics
        
        print(f"✅ RF Model training complete!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.2f} kW")
        print(f"   MAE: {mae:.2f} kW")
        
        return metrics
    
    def _create_lstm_dataset(self, series, look_back: int):
        """Create (X, y) sequences for LSTM"""
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)

    def train_lstm_model(self, look_back: int = 24):
        """Train an LSTM on the solar_kw time series."""
        if not self.data_loaded or self.df is None:
            return {"error": "No data loaded"}

        if "solar_kw" not in self.df.columns:
            return {"error": "solar_kw column missing"}

        print(f"\n🧠 Training LSTM model with sequence length: {look_back}")
        
        values = self.df["solar_kw"].values.reshape(-1, 1)
        scaled = self.lstm_scaler.fit_transform(values)

        X, y = self._create_lstm_dataset(scaled, look_back)
        if len(X) < 10:
            return {"error": "Not enough data for LSTM"}

        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

        print("Training LSTM model... (this may take a minute)")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[es],
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0)

        y_test_inv = self.lstm_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_pred_inv = self.lstm_scaler.inverse_transform(y_pred).ravel()

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
        r2 = float(r2_score(y_test_inv, y_pred_inv))

        self.lstm_model = model
        self.lstm_trained = True
        self.sequence_length = look_back

        self.lstm_metrics = {
            "model_type": "LSTM",
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "look_back": int(look_back),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

        print(f"✅ LSTM training complete!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.2f} kW")
        print(f"   MAE: {mae:.2f} kW")

        return self.lstm_metrics

    def forecast_lstm(self, hours: int = 24):
        """Multi-step forecast using trained LSTM."""
        if not self.lstm_trained or self.lstm_model is None:
            return {"error": "LSTM model not trained"}

        values = self.df["solar_kw"].values.reshape(-1, 1)
        scaled = self.lstm_scaler.transform(values)

        look_back = self.sequence_length
        current_seq = scaled[-look_back:].copy()
        preds_scaled = []

        steps = min(hours, 48)
        for _ in range(steps):
            x = current_seq.reshape((1, look_back, 1))
            y_pred = self.lstm_model.predict(x, verbose=0)[0, 0]
            preds_scaled.append(y_pred)
            current_seq = np.vstack([current_seq[1:], [[y_pred]]])

        preds = self.lstm_scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).ravel()

        preds_household = [float(max(0.0, p / 100.0)) for p in preds]
        return preds_household
    
    def predict_solar(self, features_dict):
        """Make prediction using trained RF model"""
        if not self.is_trained:
            return None
        
        feature_order = ['hour', 'day_of_year', 'month', 'day_of_week', 'is_weekend',
                        'wind-speed', 'humidity', 'average-wind-speed-(period)',
                        'average-pressure-(period)', 'temperature']
        
        available_features = [f for f in feature_order if f in features_dict]
        features_list = [features_dict[f] for f in available_features]
        
        features_scaled = self.scaler.transform([features_list])
        prediction = self.solar_model.predict(features_scaled)[0]
        
        return max(0, prediction)
    
    def get_data_stats(self):
        """Get statistics about the loaded dataset"""
        if not self.data_loaded:
            return {"error": "No data loaded"}
        
        stats = {
            'total_rows': int(len(self.df)),
            'date_range': {
                'start': self.df['Datetime'].min().isoformat() if 'Datetime' in self.df.columns else None,
                'end': self.df['Datetime'].max().isoformat() if 'Datetime' in self.df.columns else None
                },
                'solar_stats': {
                    'mean_kw': float(self.df['solar_kw'].mean()),
                    'max_kw': float(self.df['solar_kw'].max()),
                    'min_kw': float(self.df['solar_kw'].min()),
                    'std_kw': float(self.df['solar_kw'].std())
                    },
                    'weather_stats': {
                        'avg_temperature': float(self.df['temperature'].mean()),
                        'avg_humidity': float(self.df['humidity'].mean()),
                        'avg_wind_speed': float(self.df['wind-speed'].mean()),
                        'avg_pressure': float(self.df['average-pressure-(period)'].mean())
                        }
                        }
        return stats

# Initialize ML manager
ml_manager = RealDataMLManager()

class SolarDataGenerator:
    def __init__(self, ml_manager):
        self.ml_manager = ml_manager
        
    def get_current_weather_features(self):
        """Generate current weather features"""
        current_time = datetime.now()
        
        if self.ml_manager.data_loaded:
            df = self.ml_manager.df
            current_month = current_time.month
            seasonal_data = df[df['month'] == current_month]
            
            if len(seasonal_data) > 0:
                temp_mean = seasonal_data['temperature'].mean()
                temp_std = seasonal_data['temperature'].std()
                humidity_mean = seasonal_data['humidity'].mean()
                humidity_std = seasonal_data['humidity'].std()
                wind_mean = seasonal_data['wind-speed'].mean()
                wind_std = seasonal_data['wind-speed'].std()
                pressure_mean = seasonal_data['average-pressure-(period)'].mean()
                pressure_std = seasonal_data['average-pressure-(period)'].std()
            else:
                temp_mean, temp_std = df['temperature'].mean(), df['temperature'].std()
                humidity_mean, humidity_std = df['humidity'].mean(), df['humidity'].std()
                wind_mean, wind_std = df['wind-speed'].mean(), df['wind-speed'].std()
                pressure_mean, pressure_std = df['average-pressure-(period)'].mean(), df['average-pressure-(period)'].std()
            
            temperature = np.random.normal(temp_mean, temp_std * 0.3)
            humidity = np.clip(np.random.normal(humidity_mean, humidity_std * 0.3), 0, 100)
            wind_speed = max(0, np.random.normal(wind_mean, wind_std * 0.3))
            avg_wind_speed = wind_speed * (0.9 + np.random.random() * 0.2)
            pressure = np.random.normal(pressure_mean, pressure_std * 0.2)
        else:
            temperature = 58.5 + np.random.normal(0, 10)
            humidity = np.clip(np.random.normal(73.5, 15), 0, 100)
            wind_speed = max(0, np.random.normal(10.1, 3))
            avg_wind_speed = wind_speed * 0.95
            pressure = np.random.normal(30.0, 1)
        
        return {
            'hour': current_time.hour,
            'day_of_year': current_time.timetuple().tm_yday,
            'month': current_time.month,
            'day_of_week': current_time.weekday(),
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            'wind-speed': wind_speed,
            'humidity': humidity,
            'average-wind-speed-(period)': avg_wind_speed,
            'average-pressure-(period)': pressure,
            'temperature': temperature
        }
        
    def generate_realistic_data(self):
        """Generate current data using real ML model if trained"""
        current_time = datetime.now()
        weather_features = self.get_current_weather_features()
        
        if self.ml_manager.is_trained:
            solar_generation = self.ml_manager.predict_solar(weather_features)
            solar_generation = solar_generation / 100
        else:
            hour = current_time.hour
            if 6 <= hour <= 18:
                solar_curve = np.sin((hour - 6) * np.pi / 12)
                solar_generation = 8 * solar_curve + np.random.normal(0, 0.5)
            else:
                solar_generation = 0
        
        battery_level = max(10, min(100, 60 + np.random.normal(0, 20)))
        consumption = max(0.5, 3.0 + np.random.normal(0, 1.0))
        grid_export = max(0, solar_generation - consumption + np.random.normal(0, 0.2))
        
        return {
            "timestamp": current_time.isoformat(),
            "solar_generation": round(max(0, solar_generation), 2),
            "battery_level": round(battery_level, 1),
            "consumption": round(consumption, 2),
            "grid_export": round(grid_export, 2),
            "weather": {
                "temperature": round(weather_features['temperature'], 1),
                "humidity": round(weather_features['humidity'], 1),
                "wind_speed": round(weather_features['wind-speed'], 1),
                "pressure": round(weather_features['average-pressure-(period)'], 2)
            },
            "using_real_data_model": self.ml_manager.is_trained
        }

data_generator = SolarDataGenerator(ml_manager)

def generate_ai_advice(
    forecast: list,
    battery_level: float | None = None,
    avg_consumption: float | None = None,
) -> str:
    """
    Generate human-readable AI advice based on forecast data.
    This is your 'Gen-AI style' explanation layer on top of RF+LSTM.
    """
    if not forecast:
        return "I couldn't generate a forecast, so I can't give advice right now."

    preds = [f["predicted_generation"] for f in forecast]
    avg_gen = sum(preds) / len(preds)
    max_gen = max(preds)
    min_gen = min(preds)

    # Peak hours (first 24 hours only)
    peak_hours = []
    for f in forecast[:24]:
        if f["predicted_generation"] > avg_gen * 1.2:
            hour_label = datetime.fromisoformat(f["timestamp"]).strftime("%I %p")
            peak_hours.append(hour_label)

    low_hours = []
    for f in forecast[:24]:
        if f["predicted_generation"] < avg_gen * 0.5:
            hour_label = datetime.fromisoformat(f["timestamp"]).strftime("%I %p")
            low_hours.append(hour_label)

    parts = []

    # Summary
    parts.append(
        f"Over the next {len(forecast)} hours, your predicted solar generation ranges "
        f"from {min_gen:.2f} kW to {max_gen:.2f} kW, with an average of {avg_gen:.2f} kW."
    )

    # Peak usage suggestion
    if peak_hours:
        peak_str = ", ".join(peak_hours[:3])
        parts.append(
            f" Peak solar generation is expected around {peak_str}. "
            "This is the best time to run high-energy appliances "
            "(washing machine, EV charging, water heater, AC, etc.)."
        )

    # Low generation periods
    if low_hours:
        low_str = ", ".join(low_hours[:2])
        parts.append(
            f" Very low generation is expected around {low_str}. "
            "During these hours, try to avoid non-essential loads or rely on battery/grid."
        )

    # Battery logic
    if battery_level is not None:
        if battery_level < 30:
            parts.append(
                f" Your battery level is low at {battery_level:.1f}%. "
                "Prioritize charging it during high-generation hours."
            )
        elif battery_level > 80:
            parts.append(
                f" Your battery is well charged at {battery_level:.1f}%. "
                "You can safely rely on stored energy during low-generation periods."
            )

    # Consumption vs generation
    if avg_consumption is not None:
        if avg_gen > avg_consumption:
            surplus = avg_gen - avg_consumption
            parts.append(
                f" On average, solar generation ({avg_gen:.2f} kW) is higher than your "
                f"typical consumption ({avg_consumption:.2f} kW) by about {surplus:.2f} kW. "
                "Good time to schedule heavy tasks and optionally export excess to the grid."
            )
        else:
            deficit = avg_consumption - avg_gen
            parts.append(
                f" Your typical consumption ({avg_consumption:.2f} kW) is higher than "
                f"expected solar generation ({avg_gen:.2f} kW) by about {deficit:.2f} kW. "
                "Try to shift flexible loads into peak solar hours and use battery wisely."
            )

    parts.append(
        " Overall, align your heavy usage with high-generation hours and keep an eye on "
        "low-generation windows to minimize grid dependence and energy costs."
    )

    return " ".join(parts)


@app.on_event("startup")
async def startup_event():
    """Load CSV data and train models on startup"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "solar_data.csv")

    if os.path.exists(csv_path):
        success = ml_manager.load_csv_data(csv_path)
        if success:
            print("\n🌲 Training Random Forest on real data...")
            rf_metrics = ml_manager.train_solar_model()
            print("✅ RF model trained successfully!")
            
            print("\n🧠 Training LSTM on solar_kw time series...")
            lstm_metrics = ml_manager.train_lstm_model()
            if isinstance(lstm_metrics, dict) and "error" not in lstm_metrics:
                print("✅ LSTM model trained successfully!")
            else:
                print(f"⚠️  LSTM training failed: {lstm_metrics.get('error', 'Unknown error')}")
        else:
            print("Failed to load CSV data, using fallback simulation")
    else:
        print(f"CSV file not found at {csv_path}")
        print("Place your CSV file at the specified path or update the path in the code")


@app.get("/")
async def root():
    return {
        "message": "Smart Solar Energy Management System API", 
        "status": "running",
        "data_loaded": ml_manager.data_loaded,
        "ml_model_trained": ml_manager.is_trained,
        "lstm_model_trained": ml_manager.lstm_trained,
        "dataset_info": ml_manager.get_data_stats() if ml_manager.data_loaded else "No data loaded"
    }

@app.get("/api/solar/current")
async def get_current_data():
    return data_generator.generate_realistic_data()

@app.get("/api/data/stats")
async def get_dataset_stats():
    """Get statistics about your loaded dataset"""
    return ml_manager.get_data_stats()

@app.get("/api/models/performance")
async def get_model_performance():
    """Get detailed performance metrics for RF and LSTM."""
    if not ml_manager.data_loaded:
        return {"error": "No data loaded"}

    return {
        "random_forest": ml_manager.rf_metrics if ml_manager.rf_metrics else {"status": "not trained"},
        "lstm": ml_manager.lstm_metrics if ml_manager.lstm_metrics else {"status": "not trained"},
        "data": {
            "total_rows": len(ml_manager.df) if ml_manager.df is not None else 0,
            "clean_rows": len(ml_manager.df) if ml_manager.data_loaded else 0,
        },
    }

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a new CSV file for training"""
    try:
        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        success = ml_manager.load_csv_data(file_path)
        if success:
            rf_metrics = ml_manager.train_solar_model()
            lstm_metrics = ml_manager.train_lstm_model()
            return {
                "message": "File uploaded and models trained successfully", 
                "rf_metrics": rf_metrics,
                "lstm_metrics": lstm_metrics
            }
        else:
            return {"error": "Failed to process uploaded file"}
    
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.get("/api/solar/forecast/{hours}")
async def get_forecast(hours: int = 24):
    """Generate forecast for the next N hours using ML model or fallback"""
    forecast_data = []
    base_time = datetime.now()
    
    if ml_manager.is_trained and ml_manager.data_loaded:
        print(f"Using trained ML model for {hours}-hour forecast")
        for i in range(min(hours, 48)):
            future_time = base_time + timedelta(hours=i+1)
            
            if ml_manager.data_loaded:
                df = ml_manager.df
                future_month = future_time.month
                seasonal_data = df[df['month'] == future_month]
                
                if len(seasonal_data) > 0:
                    temp_base = seasonal_data['temperature'].mean()
                    humidity_base = seasonal_data['humidity'].mean()
                    wind_base = seasonal_data['wind-speed'].mean()
                    pressure_base = seasonal_data['average-pressure-(period)'].mean()
                else:
                    temp_base = df['temperature'].mean()
                    humidity_base = df['humidity'].mean()
                    wind_base = df['wind-speed'].mean()
                    pressure_base = df['average-pressure-(period)'].mean()
            else:
                temp_base, humidity_base, wind_base, pressure_base = 58.5, 73.5, 10.1, 30.0
            
            weather_features = {
                'hour': future_time.hour,
                'day_of_year': future_time.timetuple().tm_yday,
                'month': future_time.month,
                'day_of_week': future_time.weekday(),
                'is_weekend': 1 if future_time.weekday() >= 5 else 0,
                'wind-speed': max(0, wind_base + np.random.normal(0, 3)),  # Fixed bug
                'humidity': np.clip(humidity_base + np.random.normal(0, 5), 0, 100),
                'average-wind-speed-(period)': max(0, wind_base + np.random.normal(0, 2)),
                'average-pressure-(period)': pressure_base + np.random.normal(0, 0.5),
                'temperature': temp_base + np.random.normal(0, 2)
            }
            
            predicted_solar = ml_manager.predict_solar(weather_features)
            if predicted_solar is not None:
                predicted_solar = predicted_solar / 100
            else:
                predicted_solar = 0
            
            confidence = max(0.5, 0.95 - (i * 0.02))
            
            forecast_data.append({
                "timestamp": future_time.isoformat(),
                "predicted_generation": round(max(0, predicted_solar), 2),
                "confidence": round(confidence, 2),
                "weather_forecast": {
                    "temperature": round(weather_features['temperature'], 1),
                    "humidity": round(weather_features['humidity'], 1),
                    "wind_speed": round(weather_features['wind-speed'], 1),
                    "pressure": round(weather_features['average-pressure-(period)'], 2)
                }
            })
    else:
        print("ML model not trained, using simple fallback forecast")
        for i in range(min(hours, 24)):
            future_time = base_time + timedelta(hours=i+1)
            hour = future_time.hour
            
            if 6 <= hour <= 18:
                solar_curve = np.sin((hour - 6) * np.pi / 12)
                base_generation = 6 * solar_curve
                predicted_generation = max(0, base_generation + np.random.uniform(-1, 1))
            else:
                predicted_generation = 0
            
            forecast_data.append({
                "timestamp": future_time.isoformat(),
                "predicted_generation": round(predicted_generation, 2),
                "confidence": round(np.random.uniform(0.7, 0.9), 2)
            })
    
    return {
        "forecast": forecast_data, 
        "model": "Random Forest" if ml_manager.is_trained else "Simple Fallback",
        "total_predictions": len(forecast_data),
        "using_real_data": ml_manager.is_trained
    }


@app.get("/api/solar/forecast/lstm/{hours}")
async def get_lstm_forecast(hours: int = 24):
    """LSTM-only forecast for the next N hours."""
    result = ml_manager.forecast_lstm(hours)
    if isinstance(result, dict) and "error" in result:
        return result

    base_time = datetime.now()
    forecast = []
    for i, pred in enumerate(result):
        ts = base_time + timedelta(hours=i+1)
        forecast.append({
            "timestamp": ts.isoformat(),
            "predicted_generation": round(pred, 2),
            "confidence": round(max(0.5, 0.95 - (i * 0.01)), 2)
        })

    return {
        "forecast": forecast,
        "model": "LSTM",
        "total_predictions": len(forecast),
        "using_real_data": ml_manager.data_loaded,
    }


@app.get("/api/solar/forecast/hybrid/{hours}")
async def get_hybrid_forecast(hours: int = 24):
    """
    Hybrid forecast: weighted average of Random Forest and LSTM predictions.
    RF weight: 0.4 (better for weather patterns)
    LSTM weight: 0.6 (better for time series trends)
    """
    rf_response = await get_forecast(hours)
    rf_list = rf_response.get("forecast", [])

    lstm_values = ml_manager.forecast_lstm(hours)
    if isinstance(lstm_values, dict) and "error" in lstm_values:
        return {
            "error": "LSTM not available, falling back to RF only",
            "fallback": rf_response,
        }

    n = min(len(rf_list), len(lstm_values))
    hybrid_forecast = []
    
    for i in range(n):
        rf_pred = rf_list[i]["predicted_generation"]
        lstm_pred = lstm_values[i]
        
        # Weighted average: RF 40%, LSTM 60%
        hybrid_pred = 0.4 * rf_pred + 0.6 * lstm_pred

        item = dict(rf_list[i])
        item["predicted_generation"] = round(max(0.0, hybrid_pred), 2)
        item["model_breakdown"] = {
            "rf": round(rf_pred, 2),
            "lstm": round(lstm_pred, 2),
            "hybrid": round(hybrid_pred, 2)
        }
        hybrid_forecast.append(item)

    return {
        "forecast": hybrid_forecast,
        "model": "Hybrid_RF_LSTM",
        "weights": {"random_forest": 0.4, "lstm": 0.6},
        "description": "Combines Random Forest (weather patterns) with LSTM (temporal trends)",
        "total_predictions": len(hybrid_forecast),
        "using_real_data": ml_manager.data_loaded,
    }

@app.post("/api/solar/advice")
async def get_solar_advice(body: AdviceRequest):
    """
    Generate AI-powered energy management advice using the hybrid forecast.
    """
    hybrid_response = await get_hybrid_forecast(body.hours)

    if "error" in hybrid_response:
        return {
            "error": "Unable to generate advice without a working hybrid forecast.",
            "details": hybrid_response,
        }

    forecast = hybrid_response.get("forecast", [])

    advice_text = generate_ai_advice(
        forecast,
        battery_level=body.battery_level,
        avg_consumption=body.avg_consumption,
    )

    return {
        "advice": advice_text,
        "forecast_hours": len(forecast),
        "model_used": hybrid_response.get("model"),
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "battery_level": body.battery_level,
            "avg_consumption": body.avg_consumption,
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            if websocket.client_state.name != 'CONNECTED':
                break
                
            current_data = data_generator.generate_realistic_data()
            
            try:
                await websocket.send_text(json.dumps(current_data))
                await asyncio.sleep(5)
            except Exception as e:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌞 Smart Solar Energy Management System")
    print("   with Hybrid RF + LSTM Forecasting")
    print("="*60)
    print("\n📊 Features:")
    print("   • Random Forest for weather-based predictions")
    print("   • LSTM for time-series forecasting")
    print("   • Hybrid model combining both approaches")
    print("\n🌐 Endpoints:")
    print("   • Dashboard: http://localhost:3000")
    print("   • API Docs:  http://localhost:8000/docs")
    print("   • WebSocket: ws://localhost:8000/ws")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)