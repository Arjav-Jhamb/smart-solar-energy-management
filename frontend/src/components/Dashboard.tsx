import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

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
  weather_forecast?: {
    temperature: number;
    humidity: number;
    wind_speed: number;
    pressure: number;
  };
}

interface Props {
  currentData: SolarData | null;
  forecast: ForecastData[];
  historicalData: SolarData[];
}

const Dashboard: React.FC<Props> = ({ currentData, forecast, historicalData }) => {
  if (!currentData) {
    return (
      <div style={{ 
        padding: '20px', 
        textAlign: 'center',
        fontSize: '18px'
      }}>
        Loading solar data...
        <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
          Make sure your backend is running at http://localhost:8000
        </div>
      </div>
    );
  }

  const formatTime = (value: any): string => {
    return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDateTime = (value: any): string => {
    return new Date(value).toLocaleString();
  };

  const formatHour = (value: any): string => {
    return new Date(value).getHours() + 'h';
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <h1 style={{ color: '#2563eb', margin: 0 }}>Smart Solar Energy Management Dashboard</h1>
        {currentData.using_real_data_model && (
          <div style={{ 
            padding: '8px 16px', 
            backgroundColor: '#10b981', 
            color: 'white', 
            borderRadius: '20px',
            fontSize: '12px',
            fontWeight: 'bold'
          }}>
            ML Model Active
          </div>
        )}
      </div>
      
      {/* Current Status Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#f0f8ff', 
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: '2px solid #e0f2fe'
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#1565c0' }}>Solar Generation</h3>
          <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#1e40af', margin: 0 }}>
            {currentData.solar_generation} kW
          </p>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            {currentData.solar_generation > 0 ? '🔆 Active' : '🌙 No Generation'}
          </div>
        </div>
        
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#f0fff0', 
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: '2px solid #e8f5e8'
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#2e7d32' }}>Battery Level</h3>
          <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#059669', margin: 0 }}>
            {currentData.battery_level}%
          </p>
          <div style={{ 
            width: '100%', 
            height: '6px', 
            backgroundColor: '#e5e7eb', 
            borderRadius: '3px', 
            marginTop: '8px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: `${currentData.battery_level}%`, 
              height: '100%', 
              backgroundColor: currentData.battery_level > 50 ? '#059669' : currentData.battery_level > 20 ? '#f59e0b' : '#ef4444',
              borderRadius: '3px'
            }}></div>
          </div>
        </div>
        
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#fff8f0', 
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: '2px solid #fed7aa'
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#c2410c' }}>Consumption</h3>
          <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#d97706', margin: 0 }}>
            {currentData.consumption} kW
          </p>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            ⚡ Current Load
          </div>
        </div>
        
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#f8f0ff', 
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: '2px solid #e9d5ff'
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#7c2d92' }}>Grid Export</h3>
          <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#7c3aed', margin: 0 }}>
            {currentData.grid_export} kW
          </p>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            {currentData.grid_export > 0 ? '↗️ Exporting' : '📈 No Export'}
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '30px', marginBottom: '30px' }}>
        {/* Real-time Chart */}
        <div style={{ 
          backgroundColor: 'white', 
          padding: '25px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ marginBottom: '20px', color: '#374151' }}>Real-time Energy Flow</h2>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={historicalData.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={formatTime} 
                stroke="#6b7280"
                fontSize={12}
              />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip 
                labelFormatter={formatDateTime}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="solar_generation" 
                stroke="#3b82f6" 
                name="Solar Generation (kW)" 
                strokeWidth={3}
                dot={{ r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="consumption" 
                stroke="#10b981" 
                name="Consumption (kW)" 
                strokeWidth={3}
                dot={{ r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="grid_export" 
                stroke="#f59e0b" 
                name="Grid Export (kW)" 
                strokeWidth={2}
                strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Forecast Chart */}
        <div style={{ 
          backgroundColor: 'white', 
          padding: '25px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ marginBottom: '20px', color: '#374151' }}>24-Hour Solar Forecast</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={forecast.slice(0, 12)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={formatHour} 
                stroke="#6b7280"
                fontSize={12}
              />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip 
                labelFormatter={formatDateTime}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              />
              <Bar 
                dataKey="predicted_generation" 
                fill="#8b5cf6" 
                name="Predicted Generation (kW)"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Weather Information */}
      <div style={{ 
        marginTop: '30px', 
        padding: '25px', 
        backgroundColor: 'white', 
        borderRadius: '12px', 
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
      }}>
        <h2 style={{ marginBottom: '20px', color: '#374151' }}>Current Weather Conditions</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
          <div style={{ 
            padding: '15px', 
            backgroundColor: '#fef3c7', 
            borderRadius: '8px',
            border: '1px solid #fbbf24'
          }}>
            <strong style={{ color: '#92400e' }}>🌡️ Temperature:</strong>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#b45309' }}>
              {currentData.weather.temperature}°C
            </div>
          </div>
          <div style={{ 
            padding: '15px', 
            backgroundColor: '#dbeafe', 
            borderRadius: '8px',
            border: '1px solid #60a5fa'
          }}>
            <strong style={{ color: '#1e40af' }}>💧 Humidity:</strong>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#1d4ed8' }}>
              {currentData.weather.humidity}%
            </div>
          </div>
          <div style={{ 
            padding: '15px', 
            backgroundColor: '#d1fae5', 
            borderRadius: '8px',
            border: '1px solid #34d399'
          }}>
            <strong style={{ color: '#065f46' }}>🌪️ Wind Speed:</strong>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#047857' }}>
              {currentData.weather.wind_speed} m/s
            </div>
          </div>
          <div style={{ 
            padding: '15px', 
            backgroundColor: '#f3e8ff', 
            borderRadius: '8px',
            border: '1px solid #a78bfa'
          }}>
            <strong style={{ color: '#6b21a8' }}>📊 Pressure:</strong>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#7c3aed' }}>
              {currentData.weather.pressure} hPa
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div style={{ 
        marginTop: '20px', 
        padding: '15px', 
        backgroundColor: '#ecfdf5', 
        borderRadius: '8px', 
        border: '1px solid #22c55e',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <p style={{ margin: '0', color: '#166534', fontWeight: 'bold' }}>
          🟢 System Status: Connected • Last Update: {formatDateTime(currentData.timestamp)}
        </p>
        <div style={{ fontSize: '12px', color: '#166534' }}>
          Data Source: {currentData.using_real_data_model ? 'ML Model (Real Data)' : 'Simulation'}
        </div>
      </div>

      {/* Energy Balance Summary */}
      <div style={{ 
        marginTop: '20px', 
        padding: '20px', 
        backgroundColor: 'white', 
        borderRadius: '12px', 
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
      }}>
        <h3 style={{ marginBottom: '15px', color: '#374151' }}>Energy Balance</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '14px', marginBottom: '5px', color: '#6b7280' }}>Net Energy Flow</div>
            <div style={{ fontSize: '20px', fontWeight: 'bold' }}>
              {(currentData.solar_generation - currentData.consumption).toFixed(2)} kW
            </div>
          </div>
          <div style={{ flex: 2 }}>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              {currentData.solar_generation > currentData.consumption 
                ? '✅ Surplus - Exporting to grid/charging battery' 
                : '⚠️ Deficit - Drawing from battery/grid'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;