# schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserCreate(BaseModel):
    cardano_address: str
    public_key: str

class PrudenceConditionCreate(BaseModel):
    condition_type: str
    operator: str
    target_value: float
    data_source_id: str

class ExecutionRequestCreate(BaseModel):
    user_id: str
    agent_id: str
    conditions: List[str]  # List of condition IDs
    target_action: Dict[str, Any]
    amount_locked: int
    execution_fee: int
    expiry_days: int

class ExecutionRequestResponse(BaseModel):
    request_id: str
    status: str
    message: str

class MarketDataResponse(BaseModel):
    data_source_id: str
    value: float
    timestamp: str
    data_type: str

class TransactionResponse(BaseModel):
    transaction_hash: str
    action_type: str
    amount: int
    fee_paid: int
    executed_at: str
    request_id: str