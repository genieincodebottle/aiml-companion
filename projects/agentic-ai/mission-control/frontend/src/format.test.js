import { describe, it, expect } from 'vitest';
import { decidedBy, statusMeta, usd, eventPlane } from './format.js';

describe('format helpers', () => {
  it('maps known statuses', () => {
    expect(statusMeta('resolved').kind).toBe('ok');
    expect(statusMeta('dead_lettered').kind).toBe('bad');
    expect(statusMeta('awaiting_approval').label).toBe('Awaiting approval');
  });

  it('falls back for unknown status', () => {
    expect(statusMeta('weird').kind).toBe('neutral');
  });

  it('formats usd to 6 dp', () => {
    expect(usd(0.001234)).toBe('$0.001234');
    expect(usd(undefined)).toBe('$0.000000');
  });

  it('classifies events by plane', () => {
    expect(eventPlane('gateway_call')).toBe('gateway');
    expect(eventPlane('policy_denied')).toBe('policy');
    expect(eventPlane('approval_requested')).toBe('human');
    expect(eventPlane('saga_compensated')).toBe('reliability');
    expect(eventPlane('step_started')).toBe('runtime');
  });
});

describe('decidedBy', () => {
  it('says a parked case is waiting', () => {
    expect(decidedBy({ status: 'awaiting_approval' })).toBe('waiting for a reviewer');
  });
  it('names the reviewer', () => {
    expect(decidedBy({ decision: 'reject', decided_by: 'rex' })).toBe('rejected by rex');
  });
  it('never presents a policy approval as a person', () => {
    expect(decidedBy({ decision: 'approve', decided_by: 'policy:auto_approve' })).toMatch(/policy/);
  });
});

describe('eventPlane for the newer events', () => {
  it('puts failover on the gateway and maker-checker on the human plane', () => {
    expect(eventPlane('provider_failover')).toBe('gateway');
    expect(eventPlane('maker_checker_blocked')).toBe('human');
  });
});
