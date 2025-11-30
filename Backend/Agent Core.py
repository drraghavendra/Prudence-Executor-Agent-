# agentcore.py - PEX-A Prudence Engine & Masumi Agent Integration
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import logging
from dataclasses import dataclass
import hashlib
import hmac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pexa_agent")

@dataclass
class AgentConfig:
    """Configuration for PEX-A agent"""
    agent_id: str
    agent_secret: str
    masumi_network_url: str
    cardano_network: str
    min_execution_fee: int  # in lovelace
    max_slippage: float  # percentage
    health_check_interval: int  # seconds

@dataclass
class MarketCondition:
    """Represents a market condition to monitor"""
    condition_id: str
    data_source: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    weight: float  # importance weight for decision making
    refresh_interval: int  # seconds

@dataclass
class ExecutionDecision:
    """Decision made by the prudence engine"""
    should_execute: bool
    confidence: float
    conditions_met: List[str]
    conditions_failed: List[str]
    market_data_snapshot: Dict[str, Any]
    timestamp: datetime
    recommendation: str

class MasumiAgentClient:
    """Client for interacting with Masumi AI Agent Network"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        
    async def connect(self):
        """Establish connection to Masumi Agent Network"""
        try:
            self.session = aiohttp.ClientSession(
                base_url=self.config.masumi_network_url,
                headers={
                    'Authorization': f'Bearer {self.config.agent_secret}',
                    'X-Agent-ID': self.config.agent_id,
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            async with self.session.get('/api/v1/agent/status') as response:
                if response.status == 200:
                    self.connected = True
                    logger.info("✅ Successfully connected to Masumi Agent Network")
                else:
                    logger.error("❌ Failed to connect to Masumi Agent Network")
                    
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            self.connected = False
            
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
            self.connected = False
            logger.info("🔌 Disconnected from Masumi Agent Network")
    
    async def get_data_agents(self, data_type: str) -> List[Dict[str, Any]]:
        """Discover data agents on the network"""
        try:
            async with self.session.get(f'/api/v1/agents/discover?type={data_type}') as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('agents', [])
                else:
                    logger.warning(f"No data agents found for type: {data_type}")
                    return []
        except Exception as e:
            logger.error(f"Error discovering data agents: {str(e)}")
            return []
    
    async def purchase_market_data(self, data_agent_id: str, data_type: str, parameters: Dict) -> Dict[str, Any]:
        """Purchase market data from a data agent"""
        try:
            payload = {
                "buyer_agent_id": self.config.agent_id,
                "data_type": data_type,
                "parameters": parameters,
                "timestamp": int(time.time())
            }
            
            async with self.session.post(
                f'/api/v1/agents/{data_agent_id}/purchase',
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Purchased {data_type} data from agent {data_agent_id}")
                    return data
                else:
                    logger.error(f"❌ Failed to purchase data from {data_agent_id}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error purchasing market data: {str(e)}")
            return {}
    
    async def submit_agent_metrics(self, metrics: Dict[str, Any]):
        """Submit agent performance metrics to Masumi network"""
        try:
            async with self.session.post('/api/v1/agent/metrics', json=metrics) as response:
                if response.status == 200:
                    logger.info("✅ Submitted agent metrics")
                else:
                    logger.warning("⚠️ Failed to submit agent metrics")
        except Exception as e:
            logger.error(f"Error submitting metrics: {str(e)}")

class PrudenceEngine:
    """AI-powered decision engine for prudent execution"""
    
    def __init__(self, masumi_client: MasumiAgentClient):
        self.masumi_client = masumi_client
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.decision_history: List[ExecutionDecision] = []
        
    def add_market_condition(self, condition: MarketCondition):
        """Add a market condition to monitor"""
        self.market_conditions[condition.condition_id] = condition
        logger.info(f"📊 Added market condition: {condition.condition_id}")
    
    def remove_market_condition(self, condition_id: str):
        """Remove a market condition"""
        if condition_id in self.market_conditions:
            del self.market_conditions[condition_id]
            logger.info(f"🗑️ Removed market condition: {condition_id}")
    
    async def evaluate_conditions(self, execution_request: Dict[str, Any]) -> ExecutionDecision:
        """Evaluate all market conditions and make execution decision"""
        logger.info(f"🔍 Evaluating conditions for request: {execution_request['request_id']}")
        
        conditions_met = []
        conditions_failed = []
        market_data_snapshot = {}
        total_confidence = 0.0
        max_possible_confidence = 0.0
        
        # Gather required data types
        required_data_types = set()
        for condition in self.market_conditions.values():
            required_data_types.add(condition.data_source)
        
        # Purchase market data from Masumi network
        for data_type in required_data_types:
            data_agents = await self.masumi_client.get_data_agents(data_type)
            if data_agents:
                # Select the best data agent (could be based on reputation, price, etc.)
                selected_agent = data_agents[0]
                market_data = await self.masumi_client.purchase_market_data(
                    selected_agent['agent_id'],
                    data_type,
                    {"refresh": True}
                )
                market_data_snapshot[data_type] = market_data
            else:
                logger.warning(f"⚠️ No data agents available for {data_type}")
                market_data_snapshot[data_type] = {"value": 0, "confidence": 0}
        
        # Evaluate each condition
        for condition_id, condition in self.market_conditions.items():
            current_data = market_data_snapshot.get(condition.data_source, {})
            current_value = current_data.get('value', 0)
            data_confidence = current_data.get('confidence', 0)
            
            is_met = self._evaluate_single_condition(
                current_value, condition.threshold, condition.operator
            )
            
            if is_met:
                conditions_met.append(condition_id)
                condition_confidence = data_confidence * condition.weight
                total_confidence += condition_confidence
            else:
                conditions_failed.append(condition_id)
            
            max_possible_confidence += condition.weight
        
        # Calculate overall confidence
        overall_confidence = (total_confidence / max_possible_confidence) if max_possible_confidence > 0 else 0
        
        # Make final decision using AI-powered logic
        should_execute = await self._make_ai_decision(
            conditions_met, conditions_failed, overall_confidence, execution_request
        )
        
        decision = ExecutionDecision(
            should_execute=should_execute,
            confidence=overall_confidence,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            market_data_snapshot=market_data_snapshot,
            timestamp=datetime.utcnow(),
            recommendation=self._generate_recommendation(should_execute, overall_confidence)
        )
        
        # Store decision in history
        self.decision_history.append(decision)
        
        logger.info(f"🎯 Decision: Execute={should_execute}, Confidence={overall_confidence:.2f}")
        
        return decision
    
    def _evaluate_single_condition(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate a single market condition"""
        if operator == 'gt':
            return current_value > threshold
        elif operator == 'lt':
            return current_value < threshold
        elif operator == 'eq':
            return abs(current_value - threshold) < 0.001  # Small tolerance for floating point
        elif operator == 'gte':
            return current_value >= threshold
        elif operator == 'lte':
            return current_value <= threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    async def _make_ai_decision(self, 
                              conditions_met: List[str], 
                              conditions_failed: List[str],
                              confidence: float,
                              execution_request: Dict[str, Any]) -> bool:
        """AI-powered decision making with risk assessment"""
        
        # Basic rule-based decision making
        if len(conditions_met) == 0:
            return False
        
        # Calculate success probability based on historical data
        success_probability = self._calculate_success_probability(conditions_met)
        
        # Consider market volatility
        market_volatility = await self._assess_market_volatility()
        
        # Risk-adjusted decision
        risk_adjusted_confidence = confidence * (1 - market_volatility)
        
        # Minimum confidence threshold
        min_confidence = 0.7  # 70% confidence threshold
        
        # All conditions must be met for high-stakes operations
        if execution_request.get('amount_locked', 0) > 1000000000:  # > 1000 ADA
            min_confidence = 0.85
            if len(conditions_failed) > 0:
                return False
        
        return risk_adjusted_confidence >= min_confidence and success_probability >= 0.6
    
    def _calculate_success_probability(self, conditions_met: List[str]) -> float:
        """Calculate success probability based on historical performance of similar conditions"""
        if not self.decision_history:
            return 0.5  # Default probability
        
        # Simple implementation - in production, use ML model
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        if not recent_decisions:
            return 0.5
        
        successful_executions = [d for d in recent_decisions if d.should_execute]
        if not successful_executions:
            return 0.0
        
        # Calculate success rate of similar condition patterns
        success_rate = len(successful_executions) / len(recent_decisions)
        return success_rate
    
    async def _assess_market_volatility(self) -> float:
        """Assess current market volatility"""
        try:
            # Purchase volatility data
            volatility_agents = await self.masumi_client.get_data_agents('volatility')
            if volatility_agents:
                volatility_data = await self.masumi_client.purchase_market_data(
                    volatility_agents[0]['agent_id'],
                    'volatility',
                    {"period": "1h"}
                )
                return min(volatility_data.get('value', 0.1), 1.0)  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error assessing market volatility: {str(e)}")
        
        return 0.1  # Default low volatility
    
    def _generate_recommendation(self, should_execute: bool, confidence: float) -> str:
        """Generate human-readable recommendation"""
        if should_execute:
            if confidence > 0.9:
                return "STRONG_BUY: High confidence execution recommended"
            elif confidence > 0.7:
                return "BUY: Execution recommended with good confidence"
            else:
                return "WEAK_BUY: Execution recommended but monitor closely"
        else:
            if confidence < 0.3:
                return "STRONG_HOLD: Conditions unfavorable, avoid execution"
            else:
                return "HOLD: Wait for better market conditions"

class PEXAAgentCore:
    """Main PEX-A Agent Core that orchestrates everything"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.masumi_client = MasumiAgentClient(config)
        self.prudence_engine = PrudenceEngine(self.masumi_client)
        self.is_running = False
        self.active_requests: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the PEX-A agent"""
        logger.info("🚀 Initializing PEX-A Agent Core...")
        
        # Connect to Masumi network
        await self.masumi_client.connect()
        
        if not self.masumi_client.connected:
            logger.error("❌ Failed to initialize - cannot connect to Masumi network")
            return False
        
        # Register default market conditions
        await self._register_default_conditions()
        
        logger.info("✅ PEX-A Agent Core initialized successfully")
        return True
    
    async def _register_default_conditions(self):
        """Register default market conditions to monitor"""
        default_conditions = [
            MarketCondition(
                condition_id="min_apr",
                data_source="defi_apr",
                threshold=7.0,  # Minimum 7% APR
                operator="gte",
                weight=0.3,
                refresh_interval=300  # 5 minutes
            ),
            MarketCondition(
                condition_id="max_slippage",
                data_source="swap_slippage",
                threshold=2.0,  # Maximum 2% slippage
                operator="lte",
                weight=0.25,
                refresh_interval=60  # 1 minute
            ),
            MarketCondition(
                condition_id="liquidity_depth",
                data_source="liquidity_depth",
                threshold=100000,  # Minimum liquidity
                operator="gte",
                weight=0.2,
                refresh_interval=600  # 10 minutes
            ),
            MarketCondition(
                condition_id="market_volatility",
                data_source="volatility_index",
                threshold=0.15,  # Maximum 15% volatility
                operator="lte",
                weight=0.25,
                refresh_interval=300  # 5 minutes
            )
        ]
        
        for condition in default_conditions:
            self.prudence_engine.add_market_condition(condition)
    
    async def start_monitoring(self):
        """Start monitoring active execution requests"""
        self.is_running = True
        logger.info("👁️ Starting PEX-A monitoring service...")
        
        while self.is_running:
            try:
                # Process each active request
                for request_id, request_data in self.active_requests.items():
                    await self._process_execution_request(request_id, request_data)
                
                # Submit agent metrics periodically
                if len(self.active_requests) > 0:
                    await self._submit_performance_metrics()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self.is_running = False
        logger.info("🛑 Stopping PEX-A monitoring service...")
    
    async def add_execution_request(self, request_data: Dict[str, Any]):
        """Add a new execution request to monitor"""
        request_id = request_data['request_id']
        self.active_requests[request_id] = request_data
        logger.info(f"📥 Added execution request: {request_id}")
    
    async def remove_execution_request(self, request_id: str):
        """Remove an execution request from monitoring"""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            logger.info(f"📤 Removed execution request: {request_id}")
    
    async def _process_execution_request(self, request_id: str, request_data: Dict[str, Any]):
        """Process a single execution request"""
        try:
            # Get decision from prudence engine
            decision = await self.prudence_engine.evaluate_conditions(request_data)
            
            # Log decision
            logger.info(f"📋 Request {request_id}: Execute={decision.should_execute}, "
                       f"Confidence={decision.confidence:.2f}")
            
            # If conditions are met, trigger execution
            if decision.should_execute:
                await self._trigger_execution(request_id, request_data, decision)
            else:
                # Update request status (could notify user, etc.)
                await self._update_request_status(request_id, "waiting", decision)
                
        except Exception as e:
            logger.error(f"❌ Error processing request {request_id}: {str(e)}")
    
    async def _trigger_execution(self, request_id: str, request_data: Dict[str, Any], decision: ExecutionDecision):
        """Trigger the execution on Cardano blockchain"""
        try:
            logger.info(f"🎯 Triggering execution for request: {request_id}")
            
            # Prepare execution data for Cardano
            execution_payload = {
                "request_id": request_id,
                "decision": decision.__dict__,
                "market_data": decision.market_data_snapshot,
                "timestamp": int(time.time())
            }
            
            # In production, this would call the Cardano integration
            # For now, we'll simulate the execution
            execution_success = await self._execute_on_cardano(execution_payload)
            
            if execution_success:
                logger.info(f"✅ Successfully executed request: {request_id}")
                # Remove from active monitoring
                await self.remove_execution_request(request_id)
                
                # Record successful execution
                await self._record_successful_execution(request_id, decision)
            else:
                logger.error(f"❌ Failed to execute request: {request_id}")
                
        except Exception as e:
            logger.error(f"❌ Execution error for request {request_id}: {str(e)}")
    
    async def _execute_on_cardano(self, execution_payload: Dict[str, Any]) -> bool:
        """Execute the transaction on Cardano blockchain"""
        try:
            # This would integrate with the Cardano backend
            # For now, simulate execution
            backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{backend_url}/api/cardano/execute/{execution_payload['request_id']}",
                    json=execution_payload
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Cardano execution error: {str(e)}")
            return False
    
    async def _update_request_status(self, request_id: str, status: str, decision: ExecutionDecision):
        """Update the status of an execution request"""
        # This would update the request in the database
        # For now, just log it
        logger.info(f"📊 Request {request_id} status: {status}, "
                   f"Conditions met: {len(decision.conditions_met)}")
    
    async def _record_successful_execution(self, request_id: str, decision: ExecutionDecision):
        """Record a successful execution for analytics"""
        # This would store execution analytics
        logger.info(f"📈 Recorded successful execution: {request_id}")
    
    async def _submit_performance_metrics(self):
        """Submit performance metrics to Masumi network"""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": int(time.time()),
            "active_requests": len(self.active_requests),
            "total_decisions": len(self.prudence_engine.decision_history),
            "successful_executions": len([d for d in self.prudence_engine.decision_history 
                                        if d.should_execute]),
            "average_confidence": sum(d.confidence for d in self.prudence_engine.decision_history) 
                               / max(len(self.prudence_engine.decision_history), 1)
        }
        
        await self.masumi_client.submit_agent_metrics(metrics)

# Factory function to create and initialize the agent
async def create_pexa_agent() -> PEXAAgentCore:
    """Factory function to create and initialize PEX-A agent"""
    
    config = AgentConfig(
        agent_id=os.getenv('PEXA_AGENT_ID', 'pexa_agent_001'),
        agent_secret=os.getenv('MASUMI_AGENT_SECRET', ''),
        masumi_network_url=os.getenv('MASUMI_NETWORK_URL', 'https://api.masumi.network'),
        cardano_network=os.getenv('CARDANO_NETWORK', 'testnet'),
        min_execution_fee=5000000,  # 5 ADA
        max_slippage=2.0,  # 2%
        health_check_interval=30
    )
    
    agent = PEXAAgentCore(config)
    initialized = await agent.initialize()
    
    if initialized:
        return agent
    else:
        raise Exception("Failed to initialize PEX-A agent")

# Example usage
async def main():
    """Example of how to use the PEX-A agent"""
    try:
        # Create and initialize agent
        agent = await create_pexa_agent()
        
        # Start monitoring
        asyncio.create_task(agent.start_monitoring())
        
        # Example: Add an execution request
        sample_request = {
            "request_id": "req_12345",
            "user_id": "user_001",
            "conditions": ["min_apr", "max_slippage"],
            "target_action": {
                "action_type": "swap",
                "parameters": {
                    "from_asset": "ADA",
                    "to_asset": "MIN",
                    "amount": 100000000  # 100 ADA in lovelace
                }
            },
            "amount_locked": 150000000,  # 150 ADA
            "execution_fee": 5000000  # 5 ADA
        }
        
        await agent.add_execution_request(sample_request)
        
        # Keep the agent running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down PEX-A agent...")
        await agent.stop_monitoring()
        await agent.masumi_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())