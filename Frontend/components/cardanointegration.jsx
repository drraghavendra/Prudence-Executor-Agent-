import React from 'react';
import { useWallet } from '@meshsdk/react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Chip
} from '@mui/material';
import { CheckCircle, Error, Pending } from '@mui/icons-material';

const CardanoIntegration = ({ request, onTransactionUpdate }) => {
  const { connected, wallet } = useWallet();

  const handleSignTransaction = async (transactionCbor) => {
    try {
      const signedTx = await wallet.signTx(transactionCbor, true);
      return signedTx;
    } catch (error) {
      console.error('Transaction signing failed:', error);
      throw error;
    }
  };

  const handleSubmitTransaction = async (signedTx) => {
    try {
      const txHash = await wallet.submitTx(signedTx);
      return txHash;
    } catch (error) {
      console.error('Transaction submission failed:', error);
      throw error;
    }
  };

  const executeLockTransaction = async () => {
    try {
      // Create execution request
      const response = await fetch('http://localhost:8000/api/cardano/create-request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: request.userId,
          user_address: request.userAddress,
          conditions: request.conditions,
          target_action: request.targetAction,
          amount: request.amount,
          execution_fee: request.executionFee,
          expiry: Math.floor(Date.now() / 1000) + (30 * 24 * 60 * 60) // 30 days
        })
      });

      const result = await response.json();
      
      if (result.success) {
        // Sign and submit transaction
        const signedTx = await handleSignTransaction(result.transaction_cbor);
        const txHash = await handleSubmitTransaction(signedTx);
        
        onTransactionUpdate({
          requestId: result.request_id,
          transactionHash: txHash,
          status: 'submitted'
        });
        
        return txHash;
      }
    } catch (error) {
      console.error('Lock transaction failed:', error);
      throw error;
    }
  };

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Cardano Blockchain Integration
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Script Address:
          </Typography>
          <Chip 
            label={request.scriptAddress} 
            size="small" 
            variant="outlined"
            sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}
          />
        </Box>

        {request.transactionHash && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Transaction:
            </Typography>
            <Chip 
              label={`${request.transactionHash.slice(0, 20)}...`}
              size="small"
              icon={request.status === 'confirmed' ? <CheckCircle /> : <Pending />}
              color={request.status === 'confirmed' ? 'success' : 'warning'}
            />
          </Box>
        )}

        {!request.transactionHash && connected && (
          <Button
            variant="contained"
            onClick={executeLockTransaction}
            disabled={!request.conditions.length}
          >
            Lock Funds in Smart Contract
          </Button>
        )}

        {request.status === 'submitted' && (
          <Alert severity="info" sx={{ mt: 2 }}>
            Transaction submitted to Cardano network. Waiting for confirmation...
          </Alert>
        )}

        {request.status === 'confirmed' && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Funds successfully locked in smart contract. PEX-A will monitor conditions.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default CardanoIntegration;