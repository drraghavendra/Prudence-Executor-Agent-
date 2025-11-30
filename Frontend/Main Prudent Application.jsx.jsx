// App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { 
  Container, 
  ThemeProvider, 
  createTheme, 
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Box
} from '@mui/material';
import { blue, deepPurple } from '@mui/material/colors';

import Dashboard from './components/Dashboard';
import CreateRequest from './components/CreateRequest';
import RequestStatus from './components/RequestStatus';
import TransactionHistory from './components/TransactionHistory';
import WalletConnector from './components/WalletConnector';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: blue,
    secondary: deepPurple,
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    h4: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [user, setUser] = useState(null);
  const [walletConnected, setWalletConnected] = useState(false);

  useEffect(() => {
    // Check if user is already connected
    const savedUser = localStorage.getItem('pexa_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setWalletConnected(true);
    }
  }, []);

  const handleWalletConnect = (userData) => {
    setUser(userData);
    setWalletConnected(true);
    localStorage.setItem('pexa_user', JSON.stringify(userData));
  };

  const handleDisconnect = () => {
    setUser(null);
    setWalletConnected(false);
    localStorage.removeItem('pexa_user');
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static" elevation={0}>
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                🛡️ PEX-A - Prudent Executor Agent
              </Typography>
              <WalletConnector 
                onConnect={handleWalletConnect}
                onDisconnect={handleDisconnect}
                connected={walletConnected}
                user={user}
              />
            </Toolbar>
          </AppBar>
        </Box>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Routes>
            <Route 
              path="/" 
              element={
                walletConnected ? 
                <Dashboard user={user} /> : 
                <Navigate to="/connect" />
              } 
            />
            <Route 
              path="/create" 
              element={
                walletConnected ? 
                <CreateRequest user={user} /> : 
                <Navigate to="/connect" />
              } 
            />
            <Route 
              path="/status/:requestId" 
              element={
                walletConnected ? 
                <RequestStatus /> : 
                <Navigate to="/connect" />
              } 
            />
            <Route 
              path="/history" 
              element={
                walletConnected ? 
                <TransactionHistory user={user} /> : 
                <Navigate to="/connect" />
              } 
            />
            <Route 
              path="/connect" 
              element={
                !walletConnected ? 
                <Box 
                  sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    minHeight: '60vh'
                  }}
                >
                  <Typography variant="h4" gutterBottom>
                    Welcome to PEX-A
                  </Typography>
                  <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
                    Connect your wallet to get started with trustless DeFi automation
                  </Typography>
                  <WalletConnector 
                    onConnect={handleWalletConnect}
                    onDisconnect={handleDisconnect}
                    connected={walletConnected}
                    user={user}
                  />
                </Box> : 
                <Navigate to="/" />
              } 
            />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;