'use client'
import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, RefreshCw, Loader2, BarChart3 } from 'lucide-react'
import type { EvalRun, EvalRunDetail, PipelineComparison } from '@/types'
import {
  getEvalResults, getEvalRun, triggerEval,
  getEvalComparison, getEvalStatus,
} from '@/lib/api'
import { EvalMetricsCard } from '@/components/EvalMetricsCard'
import { PipelineComparison as PipelineComparisonChart } from '@/components/PipelineComparison'
import { QuestionDrillDown } from '@/components/QuestionDrillDown'

const METRICS_META = [
  { key: 'avg_faithfulness',      label: 'Faithfulness',      description: 'Fraction of answer claims grounded in retrieved context' },
  { key: 'avg_answer_relevance',  label: 'Answer Relevance',  description: 'How well the answer addresses the original question' },
  { key: 'avg_context_recall',    label: 'Context Recall',    description: 'Fraction of ground-truth facts found in retrieved context' },
  { key: 'avg_context_precision', label: 'Context Precision', description: 'Average precision at K — relevant docs ranked higher' },
]

export default function EvalPage() {
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [selected, setSelected] = useState<EvalRunDetail | null>(null)
  const [comparison, setComparison] = useState<PipelineComparison>({})
  const [pipeline, setPipeline] = useState<'naive' | 'crag' | 'routed'>('crag')
  const [limit, setLimit] = useState(5)
  const [running, setRunning] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [r, cmp, status] = await Promise.all([
        getEvalResults(),
        getEvalComparison(),
        getEvalStatus(),
      ])
      setRuns(r)
      setComparison(cmp)
      setRunning(status.running)
      // Auto-select latest run
      if (r.length > 0 && !selected) {
        const detail = await getEvalRun(r[0].id)
        setSelected(detail)
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [selected])

  useEffect(() => { refresh() }, [])

  // Poll while jobs are running
  useEffect(() => {
    if (!running.length) return
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [running, refresh])

  async function handleRunEval() {
    try {
      await triggerEval(pipeline, limit)
      setRunning(prev => [...new Set([...prev, pipeline])])
    } catch (e: any) {
      setError(e.message)
    }
  }

  async function handleSelectRun(run: EvalRun) {
    try {
      const detail = await getEvalRun(run.id)
      setSelected(detail)
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <div className="min-h-screen bg-[#0f1117] text-[#e2e8f0]">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-3 border-b border-[#2a2d3e] bg-[#1a1d27]">
        <Link href="/" className="flex items-center gap-1.5 text-[#64748b] hover:text-[#94a3b8] text-sm transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Chat
        </Link>
        <span className="text-[#2a2d3e]">|</span>
        <BarChart3 className="w-4 h-4 text-indigo-400" />
        <span className="font-semibold text-sm">Eval Dashboard</span>
        <div className="ml-auto flex items-center gap-2">
          {running.length > 0 && (
            <span className="flex items-center gap-1.5 text-xs text-amber-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              Running: {running.join(', ')}
            </span>
          )}
          <button
            onClick={refresh}
            className="text-[#64748b] hover:text-[#94a3b8] transition-colors p-1 rounded"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-6 space-y-8">
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {/* Run Eval Control */}
        <section className="bg-[#1a1d27] border border-[#2a2d3e] rounded-xl p-5">
          <h2 className="text-sm font-semibold mb-4 text-[#94a3b8]">Run Evaluation</h2>
          <div className="flex flex-wrap gap-3 items-end">
            <div>
              <label className="text-xs text-[#64748b] block mb-1.5">Pipeline</label>
              <div className="flex gap-1">
                {(['naive', 'crag', 'routed'] as const).map(p => (
                  <button
                    key={p}
                    onClick={() => setPipeline(p)}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                      pipeline === p
                        ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300'
                        : 'border-[#2a2d3e] text-[#64748b] hover:text-[#94a3b8]'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-xs text-[#64748b] block mb-1.5">Questions</label>
              <div className="flex gap-1">
                {[5, 10, 18].map(n => (
                  <button
                    key={n}
                    onClick={() => setLimit(n)}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                      limit === n
                        ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300'
                        : 'border-[#2a2d3e] text-[#64748b] hover:text-[#94a3b8]'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={handleRunEval}
              disabled={running.includes(pipeline)}
              className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs px-4 py-2 rounded-lg transition-colors font-medium"
            >
              {running.includes(pipeline)
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running…</>
                : <><Play className="w-3.5 h-3.5" /> Run Eval</>}
            </button>
            <p className="text-xs text-[#475569] self-end">Runs in background · results auto-refresh</p>
          </div>
        </section>

        {/* Latest metrics */}
        {selected && (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[#94a3b8]">
                Latest: <span className="text-indigo-300">{selected.pipeline}</span>
                <span className="text-[#475569] font-normal ml-2">({selected.question_count} questions)</span>
              </h2>
              <span className="text-xs text-[#475569]">{new Date(selected.timestamp).toLocaleString()}</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {METRICS_META.map(m => (
                <EvalMetricsCard
                  key={m.key}
                  metric={m.key}
                  label={m.label}
                  score={(selected as any)[m.key] ?? 0}
                  description={m.description}
                />
              ))}
            </div>
          </section>
        )}

        {/* Pipeline comparison */}
        {Object.keys(comparison).length > 0 && (
          <section className="bg-[#1a1d27] border border-[#2a2d3e] rounded-xl p-5">
            <h2 className="text-sm font-semibold mb-4 text-[#94a3b8]">Pipeline Comparison</h2>
            <PipelineComparisonChart data={comparison} />
          </section>
        )}

        {/* Run history + drill-down */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Run list */}
          <div className="bg-[#1a1d27] border border-[#2a2d3e] rounded-xl p-4">
            <h2 className="text-sm font-semibold mb-3 text-[#94a3b8]">Run History</h2>
            {loading ? (
              <p className="text-[#64748b] text-xs">Loading…</p>
            ) : runs.length === 0 ? (
              <p className="text-[#64748b] text-xs">No runs yet. Click Run Eval to start.</p>
            ) : (
              <div className="space-y-1.5">
                {runs.map(run => (
                  <button
                    key={run.id}
                    onClick={() => handleSelectRun(run)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg border text-xs transition-all ${
                      selected?.id === run.id
                        ? 'bg-indigo-600/15 border-indigo-500/30 text-indigo-300'
                        : 'border-[#2a2d3e] text-[#94a3b8] hover:border-[#3a3d4e]'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{run.pipeline}</span>
                      <span className="text-[#475569]">{Math.round(run.avg_overall * 100)}%</span>
                    </div>
                    <div className="text-[#475569] mt-0.5">
                      {run.question_count}q · {new Date(run.timestamp).toLocaleDateString()}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Drill-down */}
          <div className="lg:col-span-2 bg-[#1a1d27] border border-[#2a2d3e] rounded-xl p-4">
            <h2 className="text-sm font-semibold mb-3 text-[#94a3b8]">Question Drill-Down</h2>
            {selected?.questions?.length ? (
              <QuestionDrillDown questions={selected.questions} />
            ) : (
              <p className="text-[#64748b] text-xs">Select a run from the left to see per-question scores.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
