import { useState } from 'react';
import { COLOR } from '../lib/svg';

interface Task {
  id: string;
  title: string;
  goal: string;
  steps: {
    turn: number;
    thought: string;
    toolName: string;
    toolArgs: Record<string, any>;
    observation: string;
  }[];
  finalAnswer: string;
  expectedAnswer: string;
}

const SAMPLE_TASKS: Task[] = [
  {
    id: 'task-1',
    title: 'Task 1: H100 Roofline Knee Calculation',
    goal: 'Find the peak dense BF16 FLOP/s and HBM3 bandwidth of NVIDIA H100 SXM, then compute the arithmetic intensity roofline knee (FLOPs/byte).',
    steps: [
      {
        turn: 1,
        thought: 'I need to retrieve the official hardware specifications for NVIDIA H100 SXM.',
        toolName: 'local_search',
        toolArgs: { query: 'NVIDIA H100 SXM memory bandwidth FLOPs' },
        observation: 'Document doc_01: NVIDIA H100 SXM features 80 GB of HBM3 memory with a peak bandwidth of 3.35 TB/s and 989 TFLOP/s of dense BF16 tensor core compute.',
      },
      {
        turn: 2,
        thought: 'From doc_01, peak compute = 989e12 FLOP/s and bandwidth = 3.35e12 bytes/s. Arithmetic intensity knee = 989e12 / 3.35e12.',
        toolName: 'calculator',
        toolArgs: { expression: '989 / 3.35' },
        observation: '295.2238805970149',
      },
    ],
    finalAnswer: 'NVIDIA H100 SXM has 989 TFLOP/s dense BF16 compute and 3.35 TB/s bandwidth. The roofline arithmetic intensity knee is 989 / 3.35 ≈ 295.22 FLOPs/byte. Any kernel below 295 FLOPs/byte is memory-bound.',
    expectedAnswer: '295.22 FLOPs/byte',
  },
  {
    id: 'task-2',
    title: 'Task 2: Llama-3.1-8B KV Cache Sizing',
    goal: 'Query the architecture parameters of Llama-3.1-8B and calculate the KV cache size in bytes for one sequence of 8,192 tokens in BF16.',
    steps: [
      {
        turn: 1,
        thought: 'I will query the local technical documentation for Llama-3.1-8B layer and head counts.',
        toolName: 'local_search',
        toolArgs: { query: 'Llama-3.1-8B architecture layers heads' },
        observation: 'Document doc_06: Llama-3.1-8B model architecture: 32 transformer layers, hidden dimension 4096, 32 query heads, 8 key-value heads, head dimension 128.',
      },
      {
        turn: 2,
        thought: 'Formula: KV bytes = 2 (K and V) * layers (32) * kv_heads (8) * head_dim (128) * 2 bytes (BF16) * tokens (8192). Let us compute this in the Python sandbox.',
        toolName: 'python_sandbox',
        toolArgs: { code: 'tokens = 8192\nbytes_per_tok = 2 * 32 * 8 * 128 * 2\ntotal_bytes = tokens * bytes_per_tok\nprint(f"{total_bytes / (1024**3):.4f} GiB ({total_bytes} bytes)")' },
        observation: '1.0000 GiB (1073741824 bytes)',
      },
    ],
    finalAnswer: 'Llama-3.1-8B has 32 layers, 8 KV heads, and head dimension 128. One token occupies 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KiB). For 8,192 tokens, the KV cache requires exactly 1,073,741,824 bytes (1.0000 GiB).',
    expectedAnswer: '1.0000 GiB',
  },
  {
    id: 'task-3',
    title: 'Task 3: Memory Bandwidth Speedup Verification',
    goal: 'Compare memory bandwidth between NVIDIA T4 and NVIDIA H100 SXM, and calculate the exact speedup multiplier.',
    steps: [
      {
        turn: 1,
        thought: 'I will search for the memory bandwidth specs of NVIDIA T4 and H100.',
        toolName: 'local_search',
        toolArgs: { query: 'NVIDIA T4 memory bandwidth peak' },
        observation: 'Document doc_03: NVIDIA T4 features 16 GB of GDDR6 memory with a peak bandwidth of 320 GB/s and 65 TFLOP/s of FP16 compute.',
      },
      {
        turn: 2,
        thought: 'NVIDIA T4 bandwidth is 320 GB/s (0.32 TB/s). NVIDIA H100 bandwidth is 3.35 TB/s (3,350 GB/s). I will calculate 3350 / 320.',
        toolName: 'calculator',
        toolArgs: { expression: '3350 / 320' },
        observation: '10.46875',
      },
    ],
    finalAnswer: 'NVIDIA H100 (3,350 GB/s) has a 10.47x memory bandwidth speedup compared to NVIDIA T4 (320 GB/s).',
    expectedAnswer: '10.47x multiplier',
  },
];

export default function LabAgentSim() {
  const [activeTaskIdx, setActiveTaskIdx] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  const task = SAMPLE_TASKS[activeTaskIdx];
  const maxSteps = task.steps.length;
  const isComplete = currentStep >= maxSteps;

  const handleNextStep = () => {
    if (currentStep < maxSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
  };

  const handleSelectTask = (idx: number) => {
    setActiveTaskIdx(idx);
    setCurrentStep(0);
  };

  return (
    <div style={{ background: 'var(--ground-2)', border: '1px solid var(--rule-2)', borderRadius: 8, padding: 20, margin: '24px 0' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--token)', letterSpacing: 1, textTransform: 'uppercase' }}>
            Interactive Simulator · ReAct Loop Trace
          </span>
          <h3 style={{ margin: '4px 0 0 0', fontSize: 16, color: 'var(--text-1)' }}>
            {task.title}
          </h3>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {SAMPLE_TASKS.map((t, idx) => (
            <button
              key={t.id}
              type="button"
              className={`t-btn ${activeTaskIdx === idx ? 'active' : ''}`}
              style={{
                padding: '4px 10px',
                borderRadius: 4,
                fontSize: 11,
                background: activeTaskIdx === idx ? 'var(--ground-3)' : 'transparent',
                border: `1px solid ${activeTaskIdx === idx ? 'var(--token)' : 'var(--rule)'}`,
                color: activeTaskIdx === idx ? 'var(--token)' : 'var(--text-2)',
                cursor: 'pointer',
              }}
              onClick={() => handleSelectTask(idx)}
            >
              Task {idx + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Goal Banner */}
      <div style={{ background: 'var(--ground-3)', padding: '10px 14px', borderRadius: 6, marginBottom: 16, borderLeft: '3px solid var(--compute)' }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--compute)', textTransform: 'uppercase' }}>User Objective:</span>
        <p style={{ margin: '4px 0 0 0', fontSize: 12.5, color: 'var(--text-1)' }}>
          {task.goal}
        </p>
      </div>

      {/* Execution Stepper Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <button
          type="button"
          onClick={handleNextStep}
          disabled={isComplete}
          style={{
            padding: '6px 16px',
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 600,
            background: isComplete ? 'var(--ground-3)' : 'var(--token)',
            color: isComplete ? 'var(--text-3)' : '#0e1116',
            border: 'none',
            cursor: isComplete ? 'not-allowed' : 'pointer',
          }}
        >
          {isComplete ? 'Task Completed' : `Step Forward (Turn ${currentStep + 1})`}
        </button>
        <button
          type="button"
          onClick={handleReset}
          style={{
            padding: '6px 12px',
            borderRadius: 4,
            fontSize: 12,
            background: 'transparent',
            border: '1px solid var(--rule)',
            color: 'var(--text-2)',
            cursor: 'pointer',
          }}
        >
          Reset Trace
        </button>
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>
          Progress: {currentStep}/{maxSteps} turns executed (Step budget: 6 max)
        </span>
      </div>

      {/* Step Traces */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {task.steps.slice(0, currentStep).map((s) => (
          <div
            key={s.turn}
            style={{
              background: 'var(--ground-1)',
              border: '1px solid var(--rule-2)',
              borderRadius: 6,
              padding: 12,
              animation: 'fadeIn 0.3s ease-in-out',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 3, background: 'var(--ground-3)', color: 'var(--compute)' }}>
                TURN {s.turn}
              </span>
              <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--compute)' }}>
                Thought:
              </span>
              <span style={{ fontSize: 11.5, color: 'var(--text-1)' }}>
                {s.thought}
              </span>
            </div>

            {/* Action */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, paddingLeft: 12, borderLeft: '2px solid var(--token)' }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--token)' }}>
                Action ({s.toolName}):
              </span>
              <code style={{ fontSize: 10.5, color: 'var(--text-1)', background: 'var(--ground-3)', padding: '2px 6px', borderRadius: 3 }}>
                {JSON.stringify(s.toolArgs)}
              </code>
            </div>

            {/* Observation */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, paddingLeft: 12, borderLeft: '2px solid var(--comm)' }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--comm)', minWidth: 80 }}>
                Observation:
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-2)', fontFamily: 'monospace' }}>
                {s.observation}
              </span>
            </div>
          </div>
        ))}

        {/* Final Answer Banner if Complete */}
        {isComplete && (
          <div style={{ background: 'var(--ground-3)', border: '1px solid var(--token)', borderRadius: 6, padding: 14, marginTop: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--token)', textTransform: 'uppercase' }}>
              Final Answer Verified:
            </span>
            <p style={{ margin: '6px 0 0 0', fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.5 }}>
              {task.finalAnswer}
            </p>
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)' }}>
              Target Output: <strong style={{ color: 'var(--token)' }}>{task.expectedAnswer}</strong> · Trajectory Status: <strong style={{ color: 'var(--token)' }}>100% SUCCESS</strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
