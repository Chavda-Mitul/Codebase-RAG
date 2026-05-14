import type { AskResponse, AgentResponse, GraphStatsResponse } from '@/types'

const BASE = '/api/backend'

export async function askQuestion(question: string, topK = 8): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export async function getGraphStats(): Promise<GraphStatsResponse> {
  const res = await fetch(`${BASE}/graph/stats`)
  if (!res.ok) throw new Error('Failed to fetch graph stats')
  return res.json()
}

export async function checkHealth(): Promise<{ status: string; neo4j: boolean; tracing_enabled: boolean }> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function getEvalResults(): Promise<import('@/types').EvalRun[]> {
  const res = await fetch(`${BASE}/eval/results`)
  if (!res.ok) throw new Error('Failed to fetch eval results')
  return res.json()
}

export async function getEvalRun(runId: number): Promise<import('@/types').EvalRunDetail> {
  const res = await fetch(`${BASE}/eval/results/${runId}`)
  if (!res.ok) throw new Error(`Failed to fetch run ${runId}`)
  return res.json()
}

export async function triggerEval(pipeline: string, limit: number): Promise<{ status: string; pipeline: string; limit?: number }> {
  const res = await fetch(`${BASE}/eval/run?pipeline=${pipeline}&limit=${limit}`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to trigger eval')
  return res.json()
}

export async function getEvalComparison(): Promise<import('@/types').PipelineComparison> {
  const res = await fetch(`${BASE}/eval/compare`)
  if (!res.ok) throw new Error('Failed to fetch eval comparison')
  return res.json()
}

export async function getEvalStatus(): Promise<{ running: string[] }> {
  const res = await fetch(`${BASE}/eval/status`)
  if (!res.ok) throw new Error('Failed to fetch eval status')
  return res.json()
}

export async function agentAsk(question: string, sessionId: string): Promise<AgentResponse> {
  const res = await fetch(`${BASE}/agent/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Agent request failed')
  }
  return res.json()
}
