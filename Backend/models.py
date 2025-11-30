# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    cardano_address = Column(String, unique=True, nullable=False)
    public_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)

class PrudenceCondition(Base):
    __tablename__ = "prudence_conditions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    condition_type = Column(String, nullable=False)  # 'apr', 'price', 'volume', etc.
    operator = Column(String, nullable=False)  # '>', '<', '==', etc.
    target_value = Column(Float, nullable=False)
    data_source_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

class ExecutionRequest(Base):
    __tablename__ = "execution_requests"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)
    conditions = Column(JSON, nullable=False)  # List of condition IDs
    target_action = Column(JSON, nullable=False)  # Action parameters
    amount_locked = Column(Integer, nullable=False)  # Lovelace amount
    execution_fee = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())
    expiry = Column(DateTime, nullable=False)
    status = Column(String, default='pending')  # pending, executed, expired, cancelled
    transaction_hash = Column(String)
    datum_hash = Column(String)

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    data_source_id = Column(String, nullable=False)
    data_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    signature = Column(String)
    agent_id = Column(String)

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    agent_type = Column(String, nullable=False)  # 'executor', 'data_provider'
    public_key = Column(String, nullable=False)
    fee_structure = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

class TransactionHistory(Base):
    __tablename__ = "transaction_history"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    request_id = Column(String, nullable=False)
    transaction_hash = Column(String, nullable=False)
    action_type = Column(String, nullable=False)
    amount = Column(Integer, nullable=False)
    fee_paid = Column(Integer, nullable=False)
    executed_at = Column(DateTime, default=func.now())
    market_data_used = Column(JSON)