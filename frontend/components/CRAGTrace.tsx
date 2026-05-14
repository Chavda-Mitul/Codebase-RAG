'use client'
import { useState } from 'react'
import type { TraceInfo } from '@/types'
import clsx from 'clsx'

const ROUTE_CONFIG = {
  simple:     { label: 'SIMPLE',      color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
  complex:    { label: 'COMPLEX',     color: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
  conceptual: { label: 'CONCEPTUAL',  color: 'bg-violet-500/20 text-violet-300 border-violet-500/40' },
}

const CHECK_COLOR = {
  grounded:   'text-emerald-400',
  hallucinated: 'text-red-400',
  useful:     'text-emerald-400',
  'not useful': 'text-amber-400',
}

export function CRAGTrace({ trace }: { trace: TraceInfo }) {
  const [open, setOpen] = useState(false)
  const route = ROUTE_CONFIG[trace.route] || ROUTE_CONFIG.simple
  const relevantDocs = trace.doc_grades.filter(g => g.score === 'yes').length
  const totalDocs = trace.doc_grades.length

  return (
    <div className="rounded-lg border border-[#2a2d3e] bg-[#1a1d27] overflow-hidden">
      {/* Summary bar */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/5 transition-colors"
      >
        <span className={clsx('rounded border px-2 py-0.5 text-xs font-bold tracking-wide', route.color)}>
          {route.label}
        </span>
        <span className="text-[#64748b] text-xs">
          {trace.iterations} iteration{trace.iterations !== 1 ? 's' : ''}
          {totalDocs > 0 && ` · ${relevantDocs}/${totalDocs} docs relevant`}
          {trace.correction_triggered && ' · ⚡ correction triggered'}
        </span>
        {trace.hallucination_check && (
          <span className={clsx('text-xs ml-auto mr-2', CHECK_COLOR[trace.hallucination_check as keyof typeof CHECK_COLOR] || 'text-slate-400')}>
            {trace.hallucination_check}
          </span>
        )}
        <span className="text-[#64748b] text-xs">{open ? '▲' : '▼'}</span>
      </button>

      {/* Expanded details */}
      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-[#2a2d3e] pt-3">

          {trace.step_back_question && (
            <div>
              <p className="text-xs text-[#64748b] mb-1">Step-back question</p>
              <p className="text-sm text-[#94a3b8] italic">"{trace.step_back_question}"</p>
            </div>
          )}

          {trace.sub_questions.length > 0 && (
            <div>
              <p className="text-xs text-[#64748b] mb-1">Decomposed into {trace.sub_questions.length} sub-questions</p>
              <ol className="space-y-1">
                {trace.sub_questions.map((q, i) => (
                  <li key={i} className="text-sm text-[#94a3b8]">
                    <span className="text-[#64748b]">{i+1}. </span>{q}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {trace.query_rewrites.length > 0 && (
            <div>
              <p className="text-xs text-[#64748b] mb-1">Query rewrites</p>
              <ol className="space-y-1">
                {trace.query_rewrites.map((r, i) => (
                  <li key={i} className="text-sm text-amber-300/80 font-mono">
                    <span className="text-[#64748b]">{i+1}. </span>{r}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {trace.doc_grades.length > 0 && (
            <div>
              <p className="text-xs text-[#64748b] mb-1">Document grades</p>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {trace.doc_grades.map((g, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <span className={clsx('shrink-0 font-bold', g.score === 'yes' ? 'text-emerald-400' : 'text-red-400')}>
                      {g.score === 'yes' ? '✓' : '✗'}
                    </span>
                    <span className="text-[#94a3b8] font-mono shrink-0">{g.name}</span>
                    <span className="text-[#64748b]">{g.reason}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-4 text-xs pt-1">
            {trace.hallucination_check && (
              <span className="text-[#64748b]">
                Hallucination: <span className={CHECK_COLOR[trace.hallucination_check as keyof typeof CHECK_COLOR] || 'text-slate-400'}>
                  {trace.hallucination_check}
                </span>
              </span>
            )}
            {trace.answer_check && (
              <span className="text-[#64748b]">
                Answer quality: <span className={CHECK_COLOR[trace.answer_check as keyof typeof CHECK_COLOR] || 'text-slate-400'}>
                  {trace.answer_check}
                </span>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
