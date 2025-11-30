# main.py
import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
import aiohttp

from models import Base, User, PrudenceCondition, ExecutionRequest, MarketData, TransactionHistory
from schemas import (
    UserCreate, ExecutionRequestCreate, PrudenceConditionCreate,
    ExecutionRequestResponse
)
from cardano_integration import PexaCardanoManager, CardanoIntegration
from agentcore import create_pexa_agent, PEXAAgentCore
from pycardano import Network

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pexa_backend")

# Global instances
pexa_agent: Optional[PEXAAgentCore] = None
cardano_manager: Optional[PexaCardanoManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown events"""
    global pexa_agent, cardano_manager
    
    # Startup
    logger.info("🚀 Starting PEX-A Backend...")
    
    try:
        # Initialize Cardano manager
        cardano_manager = PexaCardanoManager()
        logger.info("✅ Cardano manager initialized")
        
        # Initialize PEX-A agent
        pexa_agent = await create_pexa_agent()
        asyncio.create_task(pexa_agent.start_monitoring())
        logger.info("✅ PEX-A Agent Core started")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PEX-A Backend...")
    
    if pexa_agent:
        await pexa_agent.stop_monitoring()
        await pexa_agent.masumi_client.disconnect()
        logger.info("✅ PEX-A Agent Core shutdown complete")

# FastAPI app with lifespan
app = FastAPI(
    title="PEX-A Backend API",
    description="Prudent Executor Agent - Trustless DeFi Automation on Cardano",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pexa.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Response models
class HealthResponse:
    status: str
    timestamp: str
    service: str
    version: str

class ErrorResponse:
    error: str
    details: Optional[str] = None
    code: Optional[str] = None

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": "HTTP_ERROR"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": "INTERNAL_ERROR"}
    )

# Utility functions
class DatabaseService:
    """Database service for common operations"""
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_execution_request(db: Session, request_id: str) -> Optional[ExecutionRequest]:
        return db.query(ExecutionRequest).filter(ExecutionRequest.id == request_id).first()
    
    @staticmethod
    def get_conditions_data(db: Session, condition_ids: List[str]) -> List[Dict[str, Any]]:
        conditions = db.query(PrudenceCondition).filter(
            PrudenceCondition.id.in_(condition_ids)
        ).all()
        
        return [
            {
                "condition_id": condition.id,
                "condition_type": condition.condition_type,
                "operator": condition.operator,
                "target_value": condition.target_value,
                "data_source_id": condition.data_source_id
            }
            for condition in conditions
        ]

class PrudenceEngine:
    """Optimized Prudence Engine with caching"""
    
    def __init__(self):
        self.masumi_agent_url = os.getenv("MASUMI_AGENT_NETWORK_URL")
        self._market_data_cache = {}
        self._cache_ttl = 60  # 1 minute cache
        
    async def evaluate_conditions(self, conditions: List[dict]) -> Dict[str, Any]:
        """Evaluate conditions with async market data fetching"""
        tasks = []
        for condition in conditions:
            task = self._fetch_condition_data(condition)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        condition_results = []
        all_satisfied = True
        
        for condition, result in zip(conditions, results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating condition {condition['condition_id']}: {str(result)}")
                satisfied = False
                current_value = 0
            else:
                current_value = result.get('value', 0)
                satisfied = self._evaluate_single_condition(
                    condition["operator"], current_value, condition["target_value"]
                )
            
            condition_results.append({
                "condition_id": condition["condition_id"],
                "condition_type": condition["condition_type"],
                "is_satisfied": satisfied,
                "actual_value": current_value,
                "required_value": condition["target_value"],
                "operator": condition["operator"]
            })
            
            if not satisfied:
                all_satisfied = False
        
        return {
            "all_satisfied": all_satisfied,
            "condition_results": condition_results,
            "evaluation_time": datetime.utcnow().isoformat()
        }
    
    async def _fetch_condition_data(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch market data for a single condition with caching"""
        source_id = condition["data_source_id"]
        cache_key = f"{source_id}_{condition['condition_type']}"
        
        # Check cache
        if cache_key in self._market_data_cache:
            cached_data, timestamp = self._market_data_cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                return cached_data
        
        # Fetch fresh data
        try:
            data = await self._fetch_from_masumi(source_id)
            self._market_data_cache[cache_key] = (data, datetime.utcnow())
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch from Masumi for {source_id}: {str(e)}")
            return await self._get_fallback_data(source_id)
    
    async def _fetch_from_masumi(self, source_id: str) -> Dict[str, Any]:
        """Fetch data from Masumi network"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.masumi_agent_url}/data/{source_id}",
                headers={"Authorization": f"Bearer {os.getenv('MASUMI_API_KEY')}"},
                timeout=10
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"HTTP {response.status}")
    
    async def _get_fallback_data(self, source_id: str) -> Dict[str, Any]:
        """Get fallback data from database"""
        db = SessionLocal()
        try:
            data = db.query(MarketData).filter(
                MarketData.data_source_id == source_id
            ).order_by(desc(MarketData.timestamp)).first()
            
            return {
                "value": data.value if data else 0,
                "timestamp": data.timestamp.isoformat() if data else datetime.utcnow().isoformat()
            }
        finally:
            db.close()
    
    def _evaluate_single_condition(self, operator: str, actual_value: float, target_value: float) -> bool:
        """Evaluate a single condition"""
        operator_map = {
            "GreaterThan": actual_value > target_value,
            "LessThan": actual_value < target_value,
            "EqualTo": abs(actual_value - target_value) < 0.001,
            "GreaterThanOrEqual": actual_value >= target_value,
            "LessThanOrEqual": actual_value <= target_value
        }
        return operator_map.get(operator, False)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, request_id: str):
        await websocket.accept()
        self.active_connections[request_id] = websocket
        logger.info(f"WebSocket connected for request {request_id}")
    
    def disconnect(self, request_id: str):
        if request_id in self.active_connections:
            del self.active_connections[request_id]
            logger.info(f"WebSocket disconnected for request {request_id}")
    
    async def send_message(self, request_id: str, message: Dict[str, Any]):
        if request_id in self.active_connections:
            try:
                await self.active_connections[request_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {str(e)}")
                self.disconnect(request_id)

ws_manager = ConnectionManager()

# API Routes
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "PEX-A Prudent Executor Agent API", "version": "2.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agent_status = "online" if pexa_agent and pexa_agent.masumi_client.connected else "offline"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "PEX-A Backend",
        "version": "2.0.0",
        "agent_status": agent_status
    }

@app.post("/users", response_model=dict)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.cardano_address == user.cardano_address).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this Cardano address already exists"
        )
    
    db_user = User(
        cardano_address=user.cardano_address,
        public_key=user.public_key
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"Created user: {db_user.id}")
    return {"user_id": db_user.id, "status": "created"}

@app.post("/prudence-conditions", response_model=dict)
async def create_prudence_condition(
    condition: PrudenceConditionCreate, 
    db: Session = Depends(get_db)
):
    """Create a new prudence condition"""
    db_condition = PrudenceCondition(
        condition_type=condition.condition_type,
        operator=condition.operator,
        target_value=condition.target_value,
        data_source_id=condition.data_source_id
    )
    db.add(db_condition)
    db.commit()
    db.refresh(db_condition)
    
    logger.info(f"Created prudence condition: {db_condition.id}")
    return {"condition_id": db_condition.id, "status": "created"}

@app.post("/execution-requests", response_model=ExecutionRequestResponse)
async def create_execution_request(
    request: ExecutionRequestCreate, 
    db: Session = Depends(get_db)
):
    """Create a new execution request"""
    # Verify user exists
    user = DatabaseService.get_user_by_id(db, request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Create execution request
    db_request = ExecutionRequest(
        user_id=request.user_id,
        agent_id=request.agent_id,
        conditions=request.conditions,
        target_action=request.target_action,
        amount_locked=request.amount_locked,
        execution_fee=request.execution_fee,
        expiry=datetime.utcnow() + timedelta(days=request.expiry_days),
        status="pending"
    )
    
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    
    logger.info(f"Created execution request: {db_request.id}")
    return {
        "request_id": db_request.id,
        "status": "created",
        "message": "Execution request created. Use Cardano endpoint to lock funds."
    }

@app.post("/cardano/lock-funds")
async def lock_funds_on_cardano(lock_request: dict, db: Session = Depends(get_db)):
    """Lock funds in Cardano smart contract"""
    execution_request = DatabaseService.get_execution_request(db, lock_request["request_id"])
    if not execution_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution request not found"
        )
    
    user = DatabaseService.get_user_by_id(db, execution_request.user_id)
    conditions_data = DatabaseService.get_conditions_data(db, execution_request.conditions)
    
    try:
        result = await cardano_manager.create_execution_request(
            user_address=user.cardano_address,
            conditions=conditions_data,
            target_action=execution_request.target_action,
            amount=execution_request.amount_locked,
            execution_fee=execution_request.execution_fee,
            expiry=int(execution_request.expiry.timestamp())
        )
        
        # Update request with Cardano data
        execution_request.datum_hash = result.get("datum_hash", "")
        db.commit()
        
        logger.info(f"Funds locked for request: {execution_request.id}")
        return {
            "success": True,
            "transaction_cbor": result["transaction_cbor"],
            "request_id": result["request_id"],
            "script_address": result["script_address"],
            "message": "Transaction ready for signing"
        }
        
    except Exception as e:
        logger.error(f"Failed to lock funds: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to lock funds: {str(e)}"
        )

@app.post("/cardano/execute/{request_id}")
async def execute_on_cardano(request_id: str, db: Session = Depends(get_db)):
    """Execute prudence conditions on Cardano blockchain"""
    request = DatabaseService.get_execution_request(db, request_id)
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found"
        )
    
    if request.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request cannot be executed"
        )
    
    # Evaluate conditions
    prudence_engine = PrudenceEngine()
    conditions_data = DatabaseService.get_conditions_data(db, request.conditions)
    
    evaluation = await prudence_engine.evaluate_conditions(conditions_data)
    
    if not evaluation["all_satisfied"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prudence conditions not satisfied"
        )
    
    try:
        # Get market data for execution
        data_sources = list(set([cond["data_source_id"] for cond in conditions_data]))
        market_data = {}
        for source_id in data_sources:
            market_data[source_id] = await prudence_engine._fetch_condition_data(
                {"data_source_id": source_id, "condition_type": "unknown"}
            )
        
        # Execute on Cardano
        result = await cardano_manager.execute_prudence_conditions(
            request_id, market_data, evaluation
        )
        
        # Update database
        request.status = "executed"
        request.transaction_hash = result["transaction_hash"]
        
        # Record transaction history
        tx_history = TransactionHistory(
            request_id=request_id,
            transaction_hash=result["transaction_hash"],
            action_type=request.target_action.get("action_type", "unknown"),
            amount=request.amount_locked,
            fee_paid=request.execution_fee,
            market_data_used=market_data
        )
        db.add(tx_history)
        db.commit()
        
        logger.info(f"Successfully executed request: {request_id}")
        return {
            "success": True,
            "transaction_hash": result["transaction_hash"],
            "execution_time": result["execution_time"],
            "message": "Execution completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Execution failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )

@app.get("/execution-requests/{request_id}/status")
async def get_request_status(request_id: str, db: Session = Depends(get_db)):
    """Get execution request status"""
    request = DatabaseService.get_execution_request(db, request_id)
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution request not found"
        )
    
    # Get transaction status
    tx_status = None
    if request.transaction_hash:
        try:
            cardano = CardanoIntegration(Network.TESTNET)
            tx_info = cardano.api.transaction(request.transaction_hash)
            tx_status = {
                "status": "confirmed" if tx_info.block else "pending",
                "block_height": tx_info.block_height,
                "block_time": tx_info.block_time
            }
        except Exception as e:
            tx_status = {"status": "error", "error": str(e)}
    
    # Evaluate current conditions
    prudence_engine = PrudenceEngine()
    conditions_data = DatabaseService.get_conditions_data(db, request.conditions)
    evaluation = await prudence_engine.evaluate_conditions(conditions_data)
    
    return {
        "request_id": request_id,
        "status": request.status,
        "conditions_met": evaluation["all_satisfied"],
        "condition_details": evaluation["condition_results"],
        "can_execute": evaluation["all_satisfied"] and request.status == "pending",
        "transaction_status": tx_status
    }

# WebSocket endpoint
@app.websocket("/ws/transaction/{request_id}")
async def websocket_transaction_updates(websocket: WebSocket, request_id: str):
    await ws_manager.connect(websocket, request_id)
    
    try:
        while True:
            # Wait for client message (ping) or timeout
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send status update
                db = SessionLocal()
                try:
                    request = DatabaseService.get_execution_request(db, request_id)
                    if request and request.transaction_hash:
                        tx_status = await get_transaction_status(request.transaction_hash)
                        await ws_manager.send_message(request_id, {
                            "type": "transaction_update",
                            "request_id": request_id,
                            "transaction_status": tx_status,
                            "execution_status": request.status
                        })
                finally:
                    db.close()
                    
    except WebSocketDisconnect:
        ws_manager.disconnect(request_id)

# Agent endpoints
@app.post("/agent/submit-request")
async def submit_agent_request(request: dict):
    """Submit execution request to PEX-A agent"""
    if not pexa_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not available"
        )
    
    try:
        await pexa_agent.add_execution_request(request)
        logger.info(f"Request submitted to agent: {request['request_id']}")
        return {
            "success": True, 
            "message": "Request submitted to PEX-A agent",
            "request_id": request["request_id"]
        }
    except Exception as e:
        logger.error(f"Failed to submit request to agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/agent/status")
async def get_agent_status():
    """Get PEX-A agent status"""
    if not pexa_agent:
        return {"status": "offline", "active_requests": 0}
    
    return {
        "status": "online",
        "active_requests": len(pexa_agent.active_requests),
        "masumi_connected": pexa_agent.masumi_client.connected,
        "decisions_made": len(pexa_agent.prudence_engine.decision_history),
        "uptime": "TODO"  # Add uptime tracking
    }

# Utility function
async def get_transaction_status(tx_hash: str):
    """Get transaction status from Cardano"""
    try:
        cardano = CardanoIntegration(Network.TESTNET)
        tx_info = cardano.api.transaction(tx_hash)
        return {
            "transaction_hash": tx_hash,
            "status": "confirmed" if tx_info.block else "pending",
            "block_height": tx_info.block_height,
            "block_time": tx_info.block_time
        }
    except Exception as e:
        return {"transaction_hash": tx_hash, "status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )