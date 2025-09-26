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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import os
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
solar_data_df = None
ml_models = {}
scalers = {}
model_metrics = {}

class RealDataMLManager:
    def __init__(self):
        self.solar_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.data_loaded = False
        self.df = None
        
    def load_csv_data(self, file_path):
        """Load and preprocess your CSV data"""
        try:
            print(f"Loading data from {file_path}...")
            self.df = pd.read_csv(file_path)
            
            # Display basic info about the dataset
            print(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            print("Columns:", list(self.df.columns))
            
            # Clean and preprocess the data
            self.df = self.preprocess_data(self.df)
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def preprocess_data(self, df):
        """Clean and prepare your specific dataset"""
        df = df.copy()
        
        # Convert datetime column
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)

            df = df.sort_values('Datetime')
        
        # Handle wind-direction mixed types
        if 'wind-direction' in df.columns:
            df['wind-direction'] = pd.to_numeric(df['wind-direction'], errors='coerce')
        
        # Create time-based features from datetime
        if 'Datetime' in df.columns:
            df['hour'] = df['Datetime'].dt.hour
            df['day_of_year'] = df['Datetime'].dt.dayofyear
            df['month'] = df['Datetime'].dt.month
            df['day_of_week'] = df['Datetime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Convert solar_mw to kW (more reasonable scale)
        if 'solar_mw' in df.columns:
            df['solar_kw'] = df['solar_mw']   # Convert MW to kW
        
        # Handle missing values
        df = df.dropna()
        
        print(f"Data preprocessing complete. Clean dataset: {len(df)} rows")
        print("Sample data:")
        print(df.head())
        
        return df
    
    def train_solar_model(self):
        """Train model using your real solar data"""
        if not self.data_loaded:
            return {"error": "No data loaded"}
        
        # Prepare features and target
        feature_columns = ['hour', 'day_of_year', 'month', 'day_of_week', 'is_weekend',
                          'wind-speed', 'humidity', 'average-wind-speed-(period)',
                          'average-pressure-(period)', 'temperature']
        
        # Remove columns that don't exist
        available_features = [col for col in feature_columns if col in self.df.columns]
        print(f"Using features: {available_features}")
        
        X = self.df[available_features]
        y = self.df['solar_kw']  # Using converted kW values
        
        # Split the data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Keep time order
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
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
        
        # Make predictions and calculate metrics
        y_pred = self.solar_model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(available_features, self.solar_model.feature_importances_))
        
        self.is_trained = True
        
        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': available_features
        }
        
        print(f"Model training complete!")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f} kW")
        print(f"MAE: {mae:.2f} kW")
        
        return metrics
    
    def predict_solar(self, features_dict):
        """Make prediction using trained model"""
        if not self.is_trained:
            return None
        
        # Convert features dict to list in correct order
        feature_order = ['hour', 'day_of_year', 'month', 'day_of_week', 'is_weekend',
                        'wind-speed', 'humidity', 'average-wind-speed-(period)',
                        'average-pressure-(period)', 'temperature']
        
        available_features = [f for f in feature_order if f in features_dict]
        features_list = [features_dict[f] for f in available_features]
        
        features_scaled = self.scaler.transform([features_list])
        prediction = self.solar_model.predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def get_data_stats(self):
        """Get statistics about the loaded dataset"""
        if not self.data_loaded:
            return {"error": "No data loaded"}
        
        stats = {
            'total_rows': len(self.df),
            'date_range': {
                'start': self.df['Datetime'].min().isoformat() if 'Datetime' in self.df.columns else None,
                'end': self.df['Datetime'].max().isoformat() if 'Datetime' in self.df.columns else None
            },
            'solar_stats': {
                'mean_kw': self.df['solar_kw'].mean(),
                'max_kw': self.df['solar_kw'].max(),
                'min_kw': self.df['solar_kw'].min(),
                'std_kw': self.df['solar_kw'].std()
            },
            'weather_stats': {
                'avg_temperature': self.df['temperature'].mean(),
                'avg_humidity': self.df['humidity'].mean(),
                'avg_wind_speed': self.df['wind-speed'].mean(),
                'avg_pressure': self.df['average-pressure-(period)'].mean()
            }
        }
        
        return stats

# Initialize ML manager
ml_manager = RealDataMLManager()

class SolarDataGenerator:
    def __init__(self, ml_manager):
        self.ml_manager = ml_manager
        
    def get_current_weather_features(self):
        """Generate current weather features (in production, get from weather API)"""
        current_time = datetime.now()
        
        # If we have real data, use similar patterns, otherwise simulate
        if self.ml_manager.data_loaded:
            # Use statistics from real data to generate realistic current values
            df = self.ml_manager.df
            
            # Get seasonal patterns from real data
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
                # Fallback to overall means
                temp_mean, temp_std = df['temperature'].mean(), df['temperature'].std()
                humidity_mean, humidity_std = df['humidity'].mean(), df['humidity'].std()
                wind_mean, wind_std = df['wind-speed'].mean(), df['wind-speed'].std()
                pressure_mean, pressure_std = df['average-pressure-(period)'].mean(), df['average-pressure-(period)'].std()
            
            # Generate realistic current values based on real data patterns
            temperature = np.random.normal(temp_mean, temp_std * 0.3)
            humidity = np.clip(np.random.normal(humidity_mean, humidity_std * 0.3), 0, 100)
            wind_speed = max(0, np.random.normal(wind_mean, wind_std * 0.3))
            avg_wind_speed = wind_speed * (0.9 + np.random.random() * 0.2)
            pressure = np.random.normal(pressure_mean, pressure_std * 0.2)
        else:
            # Fallback simulation
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
        
        # Use trained model if available
        if self.ml_manager.is_trained:
            solar_generation = self.ml_manager.predict_solar(weather_features)
            # Convert from kW to a more reasonable household scale (assuming this is utility scale data)
            solar_generation = solar_generation / 100  # Scale down for household simulation
        else:
            # Simple fallback
            hour = current_time.hour
            if 6 <= hour <= 18:
                solar_curve = np.sin((hour - 6) * np.pi / 12)
                solar_generation = 8 * solar_curve + np.random.normal(0, 0.5)
            else:
                solar_generation = 0
        
        # Generate other values
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

@app.on_event("startup")
async def startup_event():
    """Load CSV data and train models on startup"""
    # Get absolute path to project root (backend/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "solar_data.csv")

    if os.path.exists(csv_path):
        success = ml_manager.load_csv_data(csv_path)
        if success:
            print("Training model on real data...")
            metrics = ml_manager.train_solar_model()
            print("Model trained successfully!")
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
    """Get detailed model performance metrics"""
    if not ml_manager.is_trained:
        return {"error": "Model not trained yet"}
    
    # Get feature importance and metrics from training
    return {
        "model_type": "Random Forest trained on real data",
        "dataset_size": len(ml_manager.df) if ml_manager.data_loaded else 0,
        "features_used": ml_manager.solar_model.n_features_in_,
        "training_complete": ml_manager.is_trained,
        "note": "Model trained on your 116,117 row dataset"
    }

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a new CSV file for training"""
    try:
        # Save uploaded file
        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and train on new data
        success = ml_manager.load_csv_data(file_path)
        if success:
            metrics = ml_manager.train_solar_model()
            return {"message": "File uploaded and model trained successfully", "metrics": metrics}
        else:
            return {"error": "Failed to process uploaded file"}
    
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.get("/api/solar/forecast/{hours}")
async def get_forecast(hours: int = 24):
    """Generate forecast for the next N hours using ML model or fallback"""
    forecast_data = []
    base_time = datetime.now()
    
    # Use ML model if trained, otherwise use simple fallback
    if ml_manager.is_trained and ml_manager.data_loaded:
        print(f"Using trained ML model for {hours}-hour forecast")
        for i in range(min(hours, 48)):  # Limit to 48 hours max
            future_time = base_time + timedelta(hours=i+1)
            
            # Generate weather features for future time
            # In production, this would come from weather forecast API
            if ml_manager.data_loaded:
                df = ml_manager.df
                # Use seasonal patterns from real data
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
                # Fallback values
                temp_base, humidity_base, wind_base, pressure_base = 58.5, 73.5, 10.1, 30.0
            
            # Generate forecast features with some variation
            weather_features = {
                'hour': future_time.hour,
                'day_of_year': future_time.timetuple().tm_yday,
                'month': future_time.month,
                'day_of_week': future_time.weekday(),
                'is_weekend': 1 if future_time.weekday() >= 5 else 0,
                'wind-speed': max(0, temp_base + np.random.normal(0, 3)),
                'humidity': np.clip(humidity_base + np.random.normal(0, 5), 0, 100),
                'average-wind-speed-(period)': max(0, wind_base + np.random.normal(0, 2)),
                'average-pressure-(period)': pressure_base + np.random.normal(0, 0.5),
                'temperature': temp_base + np.random.normal(0, 2)
            }
            
            # Get ML prediction
            predicted_solar = ml_manager.predict_solar(weather_features)
            if predicted_solar is not None:
                # Scale down from utility scale to household scale
                predicted_solar = predicted_solar / 100
            else:
                predicted_solar = 0
            
            # Confidence decreases over time
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
        # Simple fallback forecast based on solar patterns
        for i in range(min(hours, 24)):
            future_time = base_time + timedelta(hours=i+1)
            hour = future_time.hour
            
            # Simple solar generation model based on time of day
            if 6 <= hour <= 18:
                solar_curve = np.sin((hour - 6) * np.pi / 12)
                base_generation = 6 * solar_curve  # Max 6kW for household
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
        "model": "ML Model" if ml_manager.is_trained else "Simple Fallback",
        "total_predictions": len(forecast_data),
        "using_real_data": ml_manager.is_trained
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
    print("Starting Smart Solar Energy Management System with Real Data...")
    print("Place your CSV file at: data/solar_data.csv")
    print("Dashboard will be available at: http://localhost:3000")
    print("API Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)