import { useState } from 'react';
import { Segmented, Readout } from '../lib/Stage';
import { Label, Span } from '../lib/svg';

interface TicketSample {
  id: string;
  query: string;
  baseRaw: string;
  loraRaw: string;
  expectedCategory: string;
  urgent: boolean;
}

const TICKETS: TicketSample[] = [
  {
    id: 'TCK-88192',
    query: 'EMERGENCY! Server rack TCK-88192 power supply smoking in Dallas rack 4! Shut down immediately!',
    baseRaw:
      'I can help you extract that information! Here is the JSON:\n```json\n{\n  "ticket": "TCK-88192",\n  "type": "server issue",\n  "urgent": "yes"\n}\n```\nLet me know if you need anything else!',
    loraRaw: '{"ticket_id": "TCK-88192", "category": "HARDWARE", "urgent": true}',
    expectedCategory: 'HARDWARE',
    urgent: true,
  },
  {
    id: 'TCK-11492',
    query: 'Hi team, our company was charged twice for invoice #TCK-11492 on our corporate Mastercard. Please issue credit.',
    baseRaw:
      'Sure thing! Based on your message:\n{\n  ticket_id: "TCK-11492",\n  category: "Billing / Payment",\n  urgent: false,\n}',
    loraRaw: '{"ticket_id": "TCK-11492", "category": "BILLING", "urgent": false}',
    expectedCategory: 'BILLING',
    urgent: false,
  },
  {
    id: 'TCK-39401',
    query: 'User lock-out alert: Employee on ticket TCK-39401 has reset password 5 times in 10 minutes.',
    baseRaw:
      'The extracted ticket details are as follows:\nTicket ID: TCK-39401\nCategory: User Account\nPriority: Urgent\nPlease verify employee credentials.',
    loraRaw: '{"ticket_id": "TCK-39401", "category": "ACCOUNT", "urgent": false}',
    expectedCategory: 'ACCOUNT',
    urgent: false,
  },
];

const MODES = ['Base Model (Zero-Shot)', 'LoRA Adapter (r=8)', 'Merged Weights (W0 + BA)'];

export default function LabLoraSim() {
  const [ticketIdx, setTicketIdx] = useState(0);
  const [modeIdx, setModeIdx] = useState(1); // default LoRA

  const t = TICKETS[ticketIdx];
  const isBase = modeIdx === 0;
  const isLoRA = modeIdx === 1;
  const isMerged = modeIdx === 2;

  const activeOutput = isBase ? t.baseRaw : t.loraRaw;

  let isValidJson = false;
  let parsedJson: Record<string, unknown> | null = null;
  try {
    parsedJson = JSON.parse(activeOutput);
    isValidJson = true;
  } catch {
    isValidJson = false;
  }

  return (
    <div
      style={{
        background: 'var(--canvas-elevated, #161a22)',
        border: '1px solid var(--rule-2, #30363d)',
        borderRadius: 8,
        padding: '1.25rem',
        margin: '1.5rem 0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-1)' }}>
            Interactive Lab Simulator · Qwen2.5-0.5B-Instruct
          </span>
          <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
            Evaluate format compliance before vs after LoRA fine-tuning
          </div>
        </div>

        <Segmented
          options={MODES}
          value={MODES[modeIdx]}
          onChange={(val) => {
            const idx = MODES.indexOf(val);
            if (idx !== -1) setModeIdx(idx);
          }}
        />
      </div>

      {/* Ticket Selector */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        {TICKETS.map((ticket, idx) => (
          <button
            key={ticket.id}
            type="button"
            onClick={() => setTicketIdx(idx)}
            style={{
              padding: '6px 12px',
              fontSize: 11,
              fontFamily: 'var(--mono)',
              borderRadius: 4,
              border: '1px solid',
              borderColor: ticketIdx === idx ? 'var(--text-1)' : 'var(--rule-2)',
              background: ticketIdx === idx ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
              color: ticketIdx === idx ? 'var(--text-1)' : 'var(--text-3)',
              cursor: 'pointer',
            }}
          >
            {ticket.id} ({ticket.expectedCategory})
          </button>
        ))}
      </div>

      {/* Query Card */}
      <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '0.75rem 1rem', borderRadius: 6, border: '1px solid var(--rule-2)', marginBottom: '1rem' }}>
        <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-3)', fontFamily: 'var(--mono)', marginBottom: 4 }}>
          Raw Customer Ticket Input
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-1)', fontStyle: 'italic' }}>
          &ldquo;{t.query}&rdquo;
        </div>
      </div>

      {/* Output Comparison Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        {/* Left: Model Raw Output */}
        <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '0.75rem 1rem', borderRadius: 6, border: '1px solid var(--rule-2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
              Raw Model Output
            </span>
            <span
              style={{
                fontSize: 10,
                fontFamily: 'var(--mono)',
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: 3,
                background: isValidJson ? 'rgba(18, 183, 106, 0.15)' : 'rgba(240, 68, 56, 0.15)',
                color: isValidJson ? 'var(--token)' : 'var(--error)',
                border: `1px solid ${isValidJson ? 'var(--token)' : 'var(--error)'}`,
              }}
            >
              {isValidJson ? '✓ STRICT JSON VALID' : '✗ JSON PARSE FAILED'}
            </span>
          </div>
          <pre
            style={{
              margin: 0,
              fontSize: 11.5,
              fontFamily: 'var(--mono)',
              color: isBase ? 'var(--text-2)' : 'var(--token)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.5,
              minHeight: 85,
            }}
          >
            {activeOutput}
          </pre>
        </div>

        {/* Right: Runtime & Parameter Inspector */}
        <div style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '0.75rem 1rem', borderRadius: 6, border: '1px solid var(--rule-2)' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-3)', fontFamily: 'var(--mono)', marginBottom: 8 }}>
            Engine Specs & Memory Allocation
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: 11, fontFamily: 'var(--mono)' }}>
            <div>
              <span style={{ color: 'var(--text-3)' }}>Base Params:</span>
              <div style={{ color: 'var(--text-1)', fontWeight: 600 }}>494.0 M (Frozen)</div>
            </div>
            <div>
              <span style={{ color: 'var(--text-3)' }}>Adapter Params:</span>
              <div style={{ color: isBase ? 'var(--text-3)' : 'var(--token)', fontWeight: 600 }}>
                {isBase ? '0 (0.00%)' : '1.18 M (0.24%)'}
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-3)' }}>Weights VRAM:</span>
              <div style={{ color: 'var(--text-1)', fontWeight: 600 }}>
                {isMerged ? '0.99 GB (Merged)' : isLoRA ? '0.99 GB + 2.4 MB' : '0.99 GB'}
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-3)' }}>Serving Latency:</span>
              <div style={{ color: 'var(--text-1)', fontWeight: 600 }}>
                {isBase ? '~140 ms (35 tokens)' : isLoRA ? '~48 ms (12 tokens)' : '~42 ms (Zero Added)'}
              </div>
            </div>
          </div>

          <div style={{ marginTop: '0.75rem', paddingTop: '0.5rem', borderTop: '1px solid var(--rule-2)', fontSize: 10.5, color: 'var(--text-2)' }}>
            {isBase && 'Base model is verbose: includes conversational pleasantries and markdown formatting that cause json.loads() syntax errors.'}
            {isLoRA && 'LoRA adapter forces weights to output strictly valid JSON with exact schema keys. Trainable parameter footprint: 0.24%.'}
            {isMerged && 'Adapter folded into base weights (W_merged = W0 + (α/r)BA). Single matrix multiply per layer with zero latency penalty.'}
          </div>
        </div>
      </div>
    </div>
  );
}
