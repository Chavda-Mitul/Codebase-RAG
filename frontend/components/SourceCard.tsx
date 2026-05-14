import type { SourceNode } from '@/types'
import clsx from 'clsx'

const TYPE_COLORS: Record<string, string> = {
  File:     'bg-blue-500/20 text-blue-300 border-blue-500/30',
  Class:    'bg-purple-500/20 text-purple-300 border-purple-500/30',
  Function: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  Concept:  'bg-amber-500/20 text-amber-300 border-amber-500/30',
  Module:   'bg-slate-500/20 text-slate-300 border-slate-500/30',
}

const SOURCE_BADGE: Record<string, string> = {
  vector: 'bg-indigo-500/20 text-indigo-300',
  bm25:   'bg-cyan-500/20 text-cyan-300',
  graph:  'bg-pink-500/20 text-pink-300',
  hybrid: 'bg-violet-500/20 text-violet-300',
}

export function SourceCard({ node, index }: { node: SourceNode; index: number }) {
  const typeColor = TYPE_COLORS[node.label] || TYPE_COLORS.Module
  const sourceColor = SOURCE_BADGE[node.source] || SOURCE_BADGE.hybrid

  return (
    <div className="flex items-center gap-2 rounded-lg border border-[#2a2d3e] bg-[#1a1d27] px-3 py-2 text-sm">
      <span className="text-[#64748b] font-mono text-xs w-4 shrink-0">{index}</span>
      <span className={clsx('rounded border px-1.5 py-0.5 text-xs font-medium shrink-0', typeColor)}>
        {node.label}
      </span>
      <span className="text-[#e2e8f0] font-mono truncate flex-1" title={node.name}>
        {node.name}
      </span>
      <span className={clsx('rounded px-1.5 py-0.5 text-xs shrink-0', sourceColor)}>
        {node.source}
      </span>
      <span className="text-[#64748b] text-xs shrink-0 font-mono">
        {node.score.toFixed(3)}
      </span>
    </div>
  )
}
