'use client'
import { useState } from 'react'
import { ChevronDown, ChevronRight, ChevronUp } from 'lucide-react'
import type { EvalQuestion } from '@/types'

interface Props {
  questions: EvalQuestion[]
}

type SortKey = 'faithfulness' | 'answer_relevance' | 'context_recall' | 'context_precision' | 'latency_ms'

function ScorePill({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.8 ? 'text-emerald-400' : value >= 0.6 ? 'text-amber-400' : 'text-red-400'
  return <span className={`text-xs font-mono font-medium ${color}`}>{pct}%</span>
}

export function QuestionDrillDown({ questions }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('faithfulness')
  const [sortAsc, setSortAsc] = useState(false)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const sorted = [...questions].sort((a, b) => {
    const diff = (a[sortKey] as number) - (b[sortKey] as number)
    return sortAsc ? diff : -diff
  })

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(false) }
  }

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (sortKey !== k) return null
    return sortAsc
      ? <ChevronUp className="w-3 h-3 inline ml-0.5" />
      : <ChevronDown className="w-3 h-3 inline ml-0.5" />
  }

  const cols: { key: SortKey; label: string }[] = [
    { key: 'faithfulness', label: 'Faith.' },
    { key: 'answer_relevance', label: 'Relev.' },
    { key: 'context_recall', label: 'Recall' },
    { key: 'context_precision', label: 'Prec.' },
    { key: 'latency_ms', label: 'ms' },
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[#2a2d3e] text-[#64748b]">
            <th className="text-left py-2 px-3 font-medium w-8">#</th>
            <th className="text-left py-2 px-3 font-medium">Question</th>
            {cols.map(c => (
              <th
                key={c.key}
                className="text-right py-2 px-3 font-medium cursor-pointer hover:text-[#94a3b8] whitespace-nowrap select-none"
                onClick={() => toggleSort(c.key)}
              >
                {c.label}<SortIcon k={c.key} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((q, i) => (
            <>
              <tr
                key={q.id}
                className="border-b border-[#1e2230] hover:bg-[#1a1d27] cursor-pointer transition-colors"
                onClick={() => setExpandedId(expandedId === q.id ? null : q.id)}
              >
                <td className="py-2 px-3 text-[#475569]">{i + 1}</td>
                <td className="py-2 px-3 text-[#94a3b8] max-w-[280px]">
                  <div className="flex items-center gap-1.5">
                    {expandedId === q.id
                      ? <ChevronDown className="w-3 h-3 shrink-0 text-[#64748b]" />
                      : <ChevronRight className="w-3 h-3 shrink-0 text-[#64748b]" />}
                    <span className="truncate">{q.question}</span>
                  </div>
                </td>
                <td className="py-2 px-3 text-right"><ScorePill value={q.faithfulness} /></td>
                <td className="py-2 px-3 text-right"><ScorePill value={q.answer_relevance} /></td>
                <td className="py-2 px-3 text-right"><ScorePill value={q.context_recall} /></td>
                <td className="py-2 px-3 text-right"><ScorePill value={q.context_precision} /></td>
                <td className="py-2 px-3 text-right text-[#64748b]">{Math.round(q.latency_ms)}</td>
              </tr>
              {expandedId === q.id && (
                <tr key={`${q.id}-exp`} className="bg-[#0d0f17]">
                  <td colSpan={7} className="px-4 py-3">
                    <p className="text-[#94a3b8] mb-1 font-medium text-[11px] uppercase tracking-wide">Answer</p>
                    <p className="text-[#64748b] text-xs leading-relaxed">{q.answer || '(no answer)'}</p>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}
