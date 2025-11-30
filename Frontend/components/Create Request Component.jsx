// components/CreateRequest.js
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Card,
  CardContent,
  Chip,
  Slider,
  Alert
} from '@mui/material';
import { 
  Add as AddIcon,
  Settings as ConditionsIcon,
  PlayArrow as ExecuteIcon,
  Lock as LockIcon
} from '@mui/icons-material';

const steps = ['Define Conditions', 'Set Action', 'Review & Lock'];

const conditionTypes = [
  { value: 'AprCondition', label: 'APR Condition' },
  { value: 'PriceCondition', label: 'Price Condition' },
  { value: 'VolumeCondition', label: 'Volume Condition' },
  { value: 'TimeCondition', label: 'Time Condition' }
];

const operators = [
  { value: 'GreaterThan', label: 'Greater Than (>)' },
  { value: 'LessThan', label: 'Less Than (<)' },
  { value: 'EqualTo', label: 'Equal To (==)' },
  { value: 'GreaterThanOrEqual', label: 'Greater Than or Equal (>=)' },
  { value: 'LessThanOrEqual', label: 'Less Than or Equal (<=)' }
];

const dataSources = [
  { id: 'sundaeswap_apr', name: 'SundaeSwap APR', type: 'AprData' },
  { id: 'minswap_price', name: 'MinSwap Price', type: 'PriceData' },
  { id: 'wingriders_volume', name: 'WingRiders Volume', type: 'VolumeData' },
  { id: 'blockfrost_time', name: 'Block Time', type: 'TimeData' }
];

const actionTypes = [
  { value: 'Swap', label: 'Token Swap' },
  { value: 'ProvideLiquidity', label: 'Provide Liquidity' },
  { value: 'RemoveLiquidity', label: 'Remove Liquidity' },
  { value: 'Stake', label: 'Stake Tokens' }
];

const CreateRequest = ({ user }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [conditions, setConditions] = useState([]);
  const [currentCondition, setCurrentCondition] = useState({
    condition_type: '',
    operator: '',
    target_value: 0,
    data_source_id: ''
  });
  const [targetAction, setTargetAction] = useState({
    action_type: '',
    parameters: {}
  });
  const [amount, setAmount] = useState(100);
  const [executionFee, setExecutionFee] = useState(5);

  const handleAddCondition = () => {
    if (currentCondition.condition_type && currentCondition.operator && currentCondition.data_source_id) {
      setConditions([...conditions, { ...currentCondition, id: Date.now() }]);
      setCurrentCondition({
        condition_type: '',
        operator: '',
        target_value: 0,
        data_source_id: ''
      });
    }
  };

  const handleRemoveCondition = (index) => {
    const newConditions = conditions.filter((_, i) => i !== index);
    setConditions(newConditions);
  };

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleSubmit = async () => {
    // Submit to backend
    const requestData = {
      user_id: user.userId,
      agent_id: 'pexa_agent_1',
      conditions: conditions.map(c => c.id),
      target_action: targetAction,
      amount_locked: amount * 1000000, // Convert to lovelace
      execution_fee: executionFee * 1000000,
      expiry_days: 30
    };

    try {
      const response = await fetch('http://localhost:8000/execution-requests/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json