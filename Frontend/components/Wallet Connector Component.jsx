// components/WalletConnector.js
import React, { useState } from 'react';
import {
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Box,
  Typography,
  Chip,
  Avatar
} from '@mui/material';
import { 
  AccountBalanceWallet, 
  CheckCircle,
  Cancel 
} from '@mui/icons-material';

const WalletConnector = ({ onConnect, onDisconnect, connected, user }) => {
  const [open, setOpen] = useState(false);
  const [address, setAddress] = useState('');
  const [publicKey, setPublicKey] = useState('');

  const handleConnect = () => {
    if (address && publicKey) {
      const userData = {
        cardanoAddress: address,
        publicKey: publicKey,
        userId: `user_${Date.now()}`
      };
      onConnect(userData);
      setOpen(false);
      setAddress('');
      setPublicKey('');
    }
  };

  const handleDisconnectClick = () => {
    onDisconnect();
  };

  if (connected && user) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Chip
          avatar={<Avatar>{user.cardanoAddress.slice(0, 2)}</Avatar>}
          label={`${user.cardanoAddress.slice(0, 10)}...${user.cardanoAddress.slice(-8)}`}
          variant="outlined"
          color="success"
          deleteIcon={<Cancel />}
          onDelete={handleDisconnectClick}
        />
        <CheckCircle color="success" />
      </Box>
    );
  }

  return (
    <>
      <Button
        variant="contained"
        startIcon={<AccountBalanceWallet />}
        onClick={() => setOpen(true)}
      >
        Connect Wallet
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Connect to PEX-A
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Connect your Cardano wallet to start using Prudent Executor Agent
          </Typography>
          <TextField
            fullWidth
            label="Cardano Address"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            margin="normal"
            placeholder="addr1..."
          />
          <TextField
            fullWidth
            label="Public Key"
            value={publicKey}
            onChange={(e) => setPublicKey(e.target.value)}
            margin="normal"
            placeholder="ed25519_..."
            multiline
            rows={2}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
            🔒 Your keys never leave your wallet. PEX-A operates in non-custodial mode.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleConnect} 
            variant="contained"
            disabled={!address || !publicKey}
          >
            Connect
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default WalletConnector;