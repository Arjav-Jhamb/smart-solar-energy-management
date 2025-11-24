import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Dashboard from './components/Dashboard';
import './App.css';

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

function App() {
  const [currentData, setCurrentData] = useState<SolarData | null>(null);
  const [forecast, setForecast] = useState<any[]>([]);
  const [historicalData, setHistoricalData] = useState<SolarData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('Connecting...');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef<number>(0);

  const connectWebSocket = () => {
    try {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('Connected');
        setError(null);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setCurrentData(data);
          setHistoricalData(prev => [...prev, data].slice(-50));
          setConnectionStatus('Connected - Live Data');
        } catch (err) {
          console.error('Error parsing WebSocket data:', err);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('Disconnected');
        
        // Attempt to reconnect with exponential backoff
        if (reconnectAttempts.current < 5) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++;
            console.log(`Reconnection attempt ${reconnectAttempts.current}`);
            connectWebSocket();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Connection Error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setError('Failed to establish WebSocket connection');
    }
  };

  useEffect(() => {
    console.log('App starting...');
    
    // Initial data fetch
    fetchCurrentData();
    fetchForecast();
    
    // Start WebSocket connection
    connectWebSocket();

    // Cleanup function
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const fetchCurrentData = async () => {
    try {
      console.log('Fetching current data...');
      const response = await axios.get('http://localhost:8000/api/solar/current');
      console.log('Current data received:', response.data);
      setCurrentData(response.data);
    } catch (error) {
      console.error('Error fetching current data:', error);
      // Don't set error here as WebSocket might still work
    }
  };

  const fetchForecast = async () => {
    try {
      console.log('Fetching forecast...');
      const response = await axios.get('http://localhost:8000/api/solar/forecast/24');
      console.log('Forecast received:', response.data);
      setForecast(response.data.forecast);
    } catch (error) {
      console.error('Error fetching forecast:', error);
    }
  };

  // Retry function
  const handleRetry = () => {
    setError(null);
    reconnectAttempts.current = 0;
    fetchCurrentData();
    fetchForecast();
    connectWebSocket();
  };

  // Show error only if both API and WebSocket fail
  if (error && !currentData) {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h2>Connection Error</h2>
        <p>{error}</p>
        <p>Make sure the backend is running at http://localhost:8000</p>
        <button onClick={handleRetry} style={{ padding: '10px 20px', fontSize: '16px' }}>
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className="App">
      {/* Connection Status Bar */}
      <div style={{ 
        padding: '10px', 
        backgroundColor: connectionStatus.includes('Connected') ? '#d4edda' : '#f8d7da',
        color: connectionStatus.includes('Connected') ? '#155724' : '#721c24',
        textAlign: 'center',
        fontSize: '14px',
        borderBottom: '1px solid #ddd'
      }}>
        Status: {connectionStatus}
        {connectionStatus === 'Disconnected' && (
          <button 
            onClick={handleRetry}
            style={{ marginLeft: '10px', padding: '5px 10px', fontSize: '12px' }}
          >
            Reconnect
          </button>
        )}
      </div>

      <Dashboard 
        currentData={currentData}
        forecast={forecast}
        historicalData={historicalData}
      />
    </div>
  );
}

export default App;