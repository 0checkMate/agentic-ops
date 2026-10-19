# ADR-0001: Sidecar architecture with ports and adapters

Status: Accepted

## Context
The platform adds AI agents on top of an ERP. It must be credible against a real ERP
(Odoo) while remaining deployable at zero cost, where a real Odoo instance cannot be hosted.

## Decision
- The agent service is a sidecar. The ERP stays the system of record.
- The agent service owns a separate Postgres database and never writes to the ERP's tables;
  it uses the ERP's API only.
- Agents depend on the `ErpPort` protocol, not on any ERP. `MockErp` and `OdooErp` implement it.
- One shared contract test suite runs against every implementation to prevent mock drift.
- The port has no method to confirm purchase orders. Agents can only create drafts, so the
  human-approval rule is enforced by the interface itself.

## Consequences
- Agents are testable without Odoo, and the deployed demo can run on the mock.
- Each new port method requires a contract test before any adapter implements it.
- The mock models Odoo's entities and field names but not its full behaviour.
