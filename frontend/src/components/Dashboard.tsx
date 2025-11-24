import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface SolarData {
  timestamp: string;
  solar_generation: number;
  battery_level: number;
  consumption: number;
  grid_export: number;
  weather: {
    temperature: number;
    humidity: number;
    wind_speed: number;
    pressure: number;
  };
  using_real_data_model?: boolean;
}

interface ForecastData {
  timestamp: string;
  predicted_generation: number;
  confidence: number;
  model_breakdown?: {
    rf: number;
    lstm: number;
    hybrid: number;
  };
}

interface ModelMetrics {
  model_type: string;
  r2_score: number;
  rmse: number;
  mae: number;
}

interface Props {
  currentData: SolarData | null;
  forecast: ForecastData[];
  historicalData: SolarData[];
}

const Dashboard: React.FC<Props> = ({ currentData, forecast, historicalData }) => {
  const [modelMetrics, setModelMetrics] = useState<{
    random_forest: ModelMetrics;
    lstm: ModelMetrics;
  } | null>(null);
  
  const [hybridForecast, setHybridForecast] = useState<ForecastData[]>([]);
  const [aiAdvice, setAiAdvice] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<'rf' | 'lstm' | 'hybrid'>('hybrid');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
  fetchModelMetrics();
  fetchHybridForecast();
  fetchAIAdvice(); // Only fetch once on component mount
}, []); // Empty dependency array = runs only once

  const fetchModelMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/models/performance');
      setModelMetrics(response.data);
    } catch (error) {
      console.error('Error fetching model metrics:', error);
    }
  };

  const fetchHybridForecast = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/solar/forecast/hybrid/24');
      setHybridForecast(response.data.forecast);
    } catch (error) {
      console.error('Error fetching hybrid forecast:', error);
    }
  };

  const fetchAIAdvice = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/solar/advice', {
        hours: 24,
        battery_level: currentData?.battery_level || 65,
        avg_consumption: currentData?.consumption || 3.5
      });
      setAiAdvice(response.data.advice);
    } catch (error) {
      console.error('Error fetching AI advice:', error);
      setAiAdvice('Unable to generate AI advice at this time.');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (value: any): string => {
    return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatHour = (value: any): string => {
    return new Date(value).getHours() + 'h';
  };

  if (!currentData) {
    return (
      <div style={{ 
        padding: '40px', 
        textAlign: 'center',
        fontSize: '18px',
        backgroundColor: '#f9fafb',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
        <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>Loading Solar Data...</div>
        <div style={{ fontSize: '14px', color: '#666' }}>
          Connecting to backend at http://localhost:8000
        </div>
      </div>
    );
  }

  const displayForecast = selectedModel === 'hybrid' ? hybridForecast : forecast;

  return (
    <div style={{ 
      padding: '30px', 
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', 
      backgroundColor: '#f9fafb', 
      minHeight: '100vh' 
    }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '30px',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '30px',
        borderRadius: '16px',
        color: 'white',
        boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: '0 0 10px 0', fontSize: '32px' }}>
              ☀️ Smart Solar Energy Management
            </h1>
            <p style={{ margin: 0, opacity: 0.9, fontSize: '16px' }}>
              Hybrid AI-Powered Forecasting System
            </p>
          </div>
          {currentData.using_real_data_model && (
            <div style={{ 
              padding: '12px 24px', 
              backgroundColor: 'rgba(255,255,255,0.2)', 
              backdropFilter: 'blur(10px)',
              borderRadius: '30px',
              fontSize: '14px',
              fontWeight: 'bold',
              border: '2px solid rgba(255,255,255,0.3)'
            }}>
              🤖 ML Model Active
            </div>
          )}
        </div>
      </div>

      {/* Current Status Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
        gap: '20px', 
        marginBottom: '30px' 
      }}>
        {[
          { title: 'Solar Generation', value: currentData.solar_generation, unit: 'kW', color: '#3b82f6', bg: '#dbeafe', icon: '☀️' },
          { title: 'Battery Level', value: currentData.battery_level, unit: '%', color: '#10b981', bg: '#d1fae5', icon: '🔋' },
          { title: 'Consumption', value: currentData.consumption, unit: 'kW', color: '#f59e0b', bg: '#fef3c7', icon: '⚡' },
          { title: 'Grid Export', value: currentData.grid_export, unit: 'kW', color: '#8b5cf6', bg: '#ede9fe', icon: '↗️' }
        ].map((item, idx) => (
          <div key={idx} style={{ 
            padding: '24px', 
            backgroundColor: 'white', 
            borderRadius: '16px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
            border: `2px solid ${item.bg}`,
            transition: 'transform 0.2s',
            cursor: 'pointer'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
              <div>
                <h3 style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px', fontWeight: '600' }}>
                  {item.title}
                </h3>
                <p style={{ fontSize: '36px', fontWeight: 'bold', color: item.color, margin: 0 }}>
                  {item.value}
                  <span style={{ fontSize: '20px', marginLeft: '4px' }}>{item.unit}</span>
                </p>
              </div>
              <div style={{ fontSize: '32px' }}>{item.icon}</div>
            </div>
          </div>
        ))}
      </div>

      {/* AI Advice Panel */}
      <div style={{ 
        marginBottom: '30px',
        padding: '24px',
        backgroundColor: 'white',
        borderRadius: '16px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
        border: '2px solid #fef3c7'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, color: '#374151', fontSize: '20px' }}>
            🤖 AI-Powered Energy Advice
          </h2>
          <button
            onClick={fetchAIAdvice}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#8b5cf6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              opacity: loading ? 0.6 : 1
            }}
          >
            {loading ? '⏳ Generating...' : '🔄 Refresh Advice'}
          </button>
        </div>
        <div style={{ 
          padding: '20px',
          backgroundColor: '#fffbeb',
          borderRadius: '12px',
          border: '1px solid #fbbf24',
          lineHeight: '1.8',
          fontSize: '15px',
          color: '#374151'
        }}>
          {loading ? (
            <div style={{ textAlign: 'center', color: '#6b7280' }}>
              Analyzing forecast data and generating recommendations...
            </div>
          ) : (
            aiAdvice || 'Loading AI recommendations...'
          )}
        </div>
      </div>

      {/* Model Performance Comparison */}
      {modelMetrics && (
        <div style={{ 
          marginBottom: '30px',
          padding: '24px',
          backgroundColor: 'white',
          borderRadius: '16px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.07)'
        }}>
          <h2 style={{ marginBottom: '20px', color: '#374151', fontSize: '20px' }}>
            📊 Model Performance Comparison
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            {[
              { name: 'Random Forest', data: modelMetrics.random_forest, color: '#3b82f6', icon: '🌲' },
              { name: 'LSTM', data: modelMetrics.lstm, color: '#10b981', icon: '🧠' }
            ].map((model, idx) => (
              <div key={idx} style={{ 
                padding: '20px',
                backgroundColor: '#f9fafb',
                borderRadius: '12px',
                border: `2px solid ${model.color}20`
              }}>
                <h3 style={{ margin: '0 0 16px 0', color: model.color, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{model.icon}</span>
                  {model.name}
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>R² Score:</span>
                    <span style={{ fontWeight: 'bold', color: model.color }}>
                      {(model.data.r2_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>RMSE:</span>
                    <span style={{ fontWeight: 'bold' }}>{model.data.rmse.toFixed(2)} kW</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>MAE:</span>
                    <span style={{ fontWeight: 'bold' }}>{model.data.mae.toFixed(2)} kW</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Forecast Model Selector */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', justifyContent: 'center' }}>
        {[
          { id: 'rf', label: '🌲 Random Forest', color: '#3b82f6' },
          { id: 'lstm', label: '🧠 LSTM', color: '#10b981' },
          { id: 'hybrid', label: '🤝 Hybrid (Best)', color: '#8b5cf6' }
        ].map((btn) => (
          <button
            key={btn.id}
            onClick={() => setSelectedModel(btn.id as any)}
            style={{
              padding: '12px 24px',
              backgroundColor: selectedModel === btn.id ? btn.color : 'white',
              color: selectedModel === btn.id ? 'white' : '#374151',
              border: `2px solid ${btn.color}`,
              borderRadius: '12px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              transition: 'all 0.2s'
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>

      {/* Charts Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '30px', marginBottom: '30px' }}>
        {/* Real-time Chart */}
        <div style={{ 
          backgroundColor: 'white', 
          padding: '24px', 
          borderRadius: '16px', 
          boxShadow: '0 4px 6px rgba(0,0,0,0.07)' 
        }}>
          <h2 style={{ marginBottom: '20px', color: '#374151' }}>📈 Real-time Energy Flow</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="solar_generation" stroke="#3b82f6" name="Solar (kW)" strokeWidth={3} />
              <Line type="monotone" dataKey="consumption" stroke="#10b981" name="Consumption (kW)" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Forecast Chart */}
        <div style={{ 
          backgroundColor: 'white', 
          padding: '24px', 
          borderRadius: '16px', 
          boxShadow: '0 4px 6px rgba(0,0,0,0.07)' 
        }}>
          <h2 style={{ marginBottom: '20px', color: '#374151' }}>
            🔮 24-Hour Forecast ({selectedModel.toUpperCase()})
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={displayForecast.slice(0, 12)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="timestamp" tickFormatter={formatHour} stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip />
              <Bar dataKey="predicted_generation" fill="#8b5cf6" name="Predicted (kW)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Weather Info */}
      <div style={{ 
        padding: '24px', 
        backgroundColor: 'white', 
        borderRadius: '16px', 
        boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
        marginBottom: '20px'
      }}>
        <h2 style={{ marginBottom: '20px', color: '#374151' }}>🌤️ Current Weather</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          {[
            { label: 'Temperature', value: `${currentData.weather.temperature}°C`, icon: '🌡️', color: '#f59e0b' },
            { label: 'Humidity', value: `${currentData.weather.humidity}%`, icon: '💧', color: '#3b82f6' },
            { label: 'Wind Speed', value: `${currentData.weather.wind_speed} m/s`, icon: '💨', color: '#10b981' },
            { label: 'Pressure', value: `${currentData.weather.pressure} hPa`, icon: '📊', color: '#8b5cf6' }
          ].map((item, idx) => (
            <div key={idx} style={{ 
              padding: '16px', 
              backgroundColor: '#f9fafb', 
              borderRadius: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '8px' }}>{item.icon}</div>
              <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>{item.label}</div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: item.color }}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;