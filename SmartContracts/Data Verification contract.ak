// data_verifier.ak

import aiken.{bytearray, list}

type DataAttestation {
  data_point: MarketDataPoint
  agent_id: ByteArray
  signature: ByteArray
  attestation_fee: int
}

validator data_verifier {
  datum: DataAttestation
  redeemer: ByteArray
  ctx: ScriptContext
  
  fn validate(datum: DataAttestation, redeemer: ByteArray, ctx: ScriptContext) -> Bool {
    // Verify the data agent is paid for their service
    let fee_paid = verify_attestation_fee_paid(datum.attestation_fee, datum.agent_id, ctx.transaction)
    
    // Verify the signature matches the data content
    let valid_signature = verify_data_signature(
      datum.signature,
      datum.data_point,
      datum.agent_id
    )
    
    fee_paid && valid_signature
  }
}

fn verify_attestation_fee_paid(fee: int, agent_id: ByteArray, tx: Transaction) -> Bool {
  let outputs_to_data_agent = list.filter(
    tx.outputs,
    fn(output: TransactionOutput) {
      output.address == agent_id && output.value >= fee
    }
  )
  
  list.length(outputs_to_data_agent) > 0
}

fn verify_data_signature(signature: ByteArray, data_point: MarketDataPoint, agent_id: ByteArray) -> Bool {
  // In production, use proper cryptographic verification
  // This would verify that the signature was created by the data agent
  // for the specific data content
  signature != [] && agent_id != [] && data_point.value != 0
}