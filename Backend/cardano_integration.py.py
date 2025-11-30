import os
from pycardano import *
from typing import Dict, List, Any
import json
import cbor2
from blockfrost import BlockFrostApi, ApiUrls

class CardanoIntegration:
    def __init__(self, network: Network):
        self.network = network
        self.api = BlockFrostApi(
            project_id=os.getenv("BLOCKFROST_API_KEY"),
            base_url=ApiUrls.mainnet.value if network == Network.MAINNET else ApiUrls.preprod.value
        )
        
    def build_lock_transaction(
        self, 
        user_address: Address,
        script_address: Address,
        amount: int,
        datum: Dict[str, Any],
        execution_request: Dict[str, Any]
    ) -> Transaction:
        """Build transaction to lock funds in PEX-A validator"""
        
        # Create the datum
        pexa_datum = {
            "owner": user_address.to_primitive(),
            "execution_request": execution_request,
            "amount_locked": amount,
            "created_at": execution_request["created_at"]
        }
        
        # Create script output
        script_output = TransactionOutput(
            address=script_address,
            amount=amount,
            datum=InlineDatum(pexa_datum)
        )
        
        builder = TransactionBuilder(self.network)
        builder.add_input_address(user_address)
        builder.add_output(script_output)
        
        return builder.build_and_sign([user_address])
    
    def build_execution_transaction(
        self,
        script_address: Address,
        agent_address: Address,
        redeemer: Dict[str, Any],
        market_data: List[Dict[str, Any]],
        execution_fee: int
    ) -> Transaction:
        """Build transaction to execute the prudence conditions"""
        
        # Get UTxOs at script address
        utxos = self.api.address_utxos(str(script_address))
        
        builder = TransactionBuilder(self.network)
        
        # Add script input
        for utxo in utxos:
            builder.add_script_input(
                utxo,
                script=self.load_validator_script(),
                redeemer=redeemer,
                datum=utxo.inline_datum
            )
        
        # Add outputs (return funds to user, pay fee to agent)
        total_amount = sum(utxo.amount for utxo in utxos)
        user_amount = total_amount - execution_fee
        
        builder.add_output(TransactionOutput(
            address=agent_address,
            amount=execution_fee
        ))
        
        # Return remaining to user (simplified - in reality would execute DeFi action)
        builder.add_output(TransactionOutput(
            address=Address.from_primitive(utxos[0].datum["owner"]),
            amount=user_amount
        ))
        
        return builder.build_and_sign([agent_address])
    
    def load_validator_script(self) -> PlutusV2Script:
        """Load the compiled Plutus script"""
        with open("scripts/pexa_validator.plutus", "r") as f:
            script_json = json.load(f)
        return PlutusV2Script(cbor2.loads(bytes.fromhex(script_json["cborHex"])))
    
    def submit_transaction(self, transaction: Transaction) -> str:
        """Submit transaction to Cardano network"""
        try:
            result = self.api.transaction_submit(transaction.to_cbor())
            return result
        except Exception as e:
            raise Exception(f"Transaction submission failed: {str(e)}")

# Enhanced backend integration
class PexaCardanoManager:
    def __init__(self):
        self.network = Network.TESTNET  # Change to MAINNET for production
        self.cardano = CardanoIntegration(self.network)
        self.script_address = self.load_script_address()
    
    def load_script_address(self) -> Address:
        """Load the script address from compiled validator"""
        script = self.cardano.load_validator_script()
        return Address.script_address(script, self.network)
    
    async def create_execution_request(
        self, 
        user_address: str,
        conditions: List[Dict],
        target_action: Dict,
        amount: int,
        execution_fee: int,
        expiry: int
    ) -> Dict[str, Any]:
        """Create execution request and prepare lock transaction"""
        
        execution_request = {
            "request_id": f"req_{os.urandom(8).hex()}",
            "agent_id": "pexa_agent_1",
            "conditions": conditions,
            "target_action": target_action,
            "execution_fee": execution_fee,
            "created_at": int(time.time()),
            "expiry": expiry
        }
        
        # Build lock transaction
        user_addr = Address.from_primitive(user_address)
        transaction = self.cardano.build_lock_transaction(
            user_addr,
            self.script_address,
            amount,
            {},  # Will be filled with actual datum
            execution_request
        )
        
        return {
            "execution_request": execution_request,
            "transaction_cbor": transaction.to_cbor(),
            "script_address": str(self.script_address),
            "request_id": execution_request["request_id"]
        }
    
    async def execute_prudence_conditions(
        self,
        request_id: str,
        market_data: List[Dict],
        validation_result: Dict
    ) -> Dict[str, Any]:
        """Execute prudence conditions and submit transaction"""
        
        # Build redeemer for script
        redeemer = {
            "validation_result": validation_result,
            "agent_signature": f"sig_{os.urandom(32).hex()}",  # In production, use actual signature
            "data_agent_signatures": [f"data_sig_{os.urandom(32).hex()}" for _ in market_data]
        }
        
        agent_address = Address.from_primitive(os.getenv("AGENT_ADDRESS"))
        execution_fee = 5000000  # 5 ADA in lovelace
        
        transaction = self.cardano.build_execution_transaction(
            self.script_address,
            agent_address,
            redeemer,
            market_data,
            execution_fee
        )
        
        # Submit transaction
        tx_hash = self.cardano.submit_transaction(transaction)
        
        return {
            "transaction_hash": tx_hash,
            "status": "submitted",
            "execution_time": int(time.time())
        }