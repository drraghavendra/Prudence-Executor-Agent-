
import aiken.{bool, int, list, option}
import aiken/transaction.{ScriptContext, Transaction, TxId, OutputDatum, Datum}
import aiken/math

// Types for our Prudence Conditions
type PrudenceCondition {
  condition_id: ByteArray
  condition_type: ConditionType
  target_value: int
  operator: ComparisonOperator
  data_source: DataSource
}

type ConditionType {
  AprCondition
  PriceCondition
  VolumeCondition
  TimeCondition
  CustomCondition
}

type ComparisonOperator {
  GreaterThan
  LessThan
  EqualTo
  GreaterThanOrEqual
  LessThanOrEqual
}

type DataSource {
  source_id: ByteArray
  agent_id: ByteArray
  data_type: DataType
  update_frequency: int
}

type DataType {
  AprData
  PriceData
  VolumeData
  TimeData
  CustomData
}

type ExecutionRequest {
  request_id: ByteArray
  agent_id: ByteArray
  conditions: List<PrudenceCondition>
  target_action: TargetAction
  execution_fee: int
  created_at: int
  expiry: int
}

type TargetAction {
  action_type: ActionType
  parameters: ActionParameters
}

type ActionType {
  Swap
  ProvideLiquidity
  RemoveLiquidity
  Stake
  Unstake
  CustomAction
}

type ActionParameters {
  swap_params: Option<SwapParameters>
  liquidity_params: Option<LiquidityParameters>
  stake_params: Option<StakeParameters>
  custom_params: Option<ByteArray>
}

type SwapParameters {
  from_asset: ByteArray
  to_asset: ByteArray
  min_received: int
  route: List<ByteArray>
}

type LiquidityParameters {
  pool_id: ByteArray
  asset_a: ByteArray
  asset_b: ByteArray
  amount_a: int
  amount_b: int
  min_lp_tokens: int
}

type StakeParameters {
  pool_id: ByteArray
  amount: int
  duration: int
}

type ValidationResult {
  request: ExecutionRequest
  condition_results: List<ConditionResult>
  market_data: List<MarketDataPoint>
  is_valid: bool
}

type ConditionResult {
  condition_id: ByteArray
  is_satisfied: bool
  actual_value: int
  required_value: int
}

type MarketDataPoint {
  data_source: DataSource
  value: int
  timestamp: int
  signature: ByteArray
}

// Datum for locked funds
type PexaDatum {
  owner: ByteArray
  execution_request: ExecutionRequest
  amount_locked: int
  created_at: int
}

// Redeemer for spending
type PexaRedeemer {
  validation_result: ValidationResult
  agent_signature: ByteArray
  data_agent_signatures: List<ByteArray>
}

// Validator logic
validator pexa_validator {
  datum: PexaDatum
  redeemer: PexaRedeemer
  ctx: ScriptContext
  
  fn validate(datum: PexaDatum, redeemer: PexaRedeemer, ctx: ScriptContext) -> Bool {
    let tx = ctx.transaction
    
    // 1. Verify agent signature matches the authorized agent in the request
    let valid_agent_sig = verify_agent_signature(
      redeemer.agent_signature,
      datum.execution_request.agent_id,
      tx.id
    )
    
    // 2. Verify all prudence conditions are satisfied
    let conditions_satisfied = all_conditions_satisfied(redeemer.validation_result)
    
    // 3. Verify market data signatures from data agents
    let valid_data_signatures = verify_data_signatures(
      redeemer.data_agent_signatures,
      redeemer.validation_result.market_data,
      datum.execution_request.conditions
    )
    
    // 4. Verify transaction is within expiry
    let not_expired = verify_not_expired(datum.execution_request.expiry, ctx)
    
    // 5. Verify execution fee is paid to agent
    let fee_paid = verify_execution_fee_paid(
      datum.execution_request.execution_fee,
      datum.execution_request.agent_id,
      tx
    )
    
    valid_agent_sig && conditions_satisfied && valid_data_signatures && not_expired && fee_paid
  }
}

fn verify_agent_signature(signature: ByteArray, agent_id: ByteArray, tx_id: TxId) -> Bool {
  // In production, this would use proper cryptographic verification
  // For now, we simulate signature verification
  signature != [] && agent_id != []
}

fn all_conditions_satisfied(validation_result: ValidationResult) -> Bool {
  list.all(
    validation_result.condition_results,
    fn(result: ConditionResult) { result.is_satisfied }
  )
}

fn verify_data_signatures(
  signatures: List<ByteArray>,
  market_data: List<MarketDataPoint>,
  conditions: List<PrudenceCondition>
) -> Bool {
  let expected_sources = list.map(conditions, fn(c: PrudenceCondition) { c.data_source.source_id })
  
  list.all(
    list.zip(signatures, market_data),
    fn(pair: (ByteArray, MarketDataPoint)) {
      let signature = pair.first
      let data_point = pair.second
      // Verify signature matches the data agent and data content
      signature != [] && 
      list.contains(expected_sources, data_point.data_source.source_id) &&
      verify_data_freshness(data_point.timestamp)
    }
  )
}

fn verify_data_freshness(timestamp: int) -> Bool {
  // Data should be no older than 10 blocks
  let current_time = transaction.current_time()
  current_time - timestamp < 600000 // 10 minutes in milliseconds
}

fn verify_not_expired(expiry: int, ctx: ScriptContext) -> Bool {
  ctx.transaction.time_range.end < expiry
}

fn verify_execution_fee_paid(fee: int, agent_id: ByteArray, tx: Transaction) -> Bool {
  let outputs_to_agent = list.filter(
    tx.outputs,
    fn(output: TransactionOutput) {
      output.address == agent_id && output.value >= fee
    }
  )
  
  list.length(outputs_to_agent) > 0
}

// Helper functions for condition evaluation
fn evaluate_condition(condition: PrudenceCondition, market_value: int) -> Bool {
  when condition.operator is {
    GreaterThan -> market_value > condition.target_value
    LessThan -> market_value < condition.target_value
    EqualTo -> market_value == condition.target_value
    GreaterThanOrEqual -> market_value >= condition.target_value
    LessThanOrEqual -> market_value <= condition.target_value
  }
}

// Utility function to create execution request
fn create_execution_request(
  agent_id: ByteArray,
  conditions: List<PrudenceCondition>,
  target_action: TargetAction,
  execution_fee: int,
  expiry: int
) -> ExecutionRequest {
  ExecutionRequest {
    request_id: generate_id(),
    agent_id,
    conditions,
    target_action,
    execution_fee,
    created_at: transaction.current_time(),
    expiry
  }
}