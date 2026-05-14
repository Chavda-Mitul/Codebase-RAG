'use client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts'
import type { PipelineComparison as PipelineComparisonData } from '@/types'

interface Props {
  data: PipelineComparisonData
}

const METRICS = [
  { key: 'avg_faithfulness',     label: 'Faithfulness',  color: '#6366f1' },
  { key: 'avg_answer_relevance', label: 'Relevance',     color: '#10b981' },
  { key: 'avg_context_recall',   label: 'Recall',        color: '#f59e0b' },
  { key: 'avg_context_precision',label: 'Precision',     color: '#ec4899' },
]

export function PipelineComparison({ data }: Props) {
  const pipelines = Object.keys(data)
  if (!pipelines.length) return (
    <p className="text-[#64748b] text-sm text-center py-8">No comparison data yet. Run eval on multiple pipelines.</p>
  )

  const chartData = pipelines.map(pipeline => ({
    pipeline,
    ...Object.fromEntries(
      METRICS.map(m => [m.label, Math.round((data[pipeline] as any)[m.key] * 100)])
    ),
  }))

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 4, right: 16, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3e" />
          <XAxis dataKey="pipeline" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis domain={[0, 100]} unit="%" tick={{ fill: '#64748b', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1a1d27', border: '1px solid #2a2d3e', borderRadius: 8 }}
            labelStyle={{ color: '#e2e8f0' }}
            itemStyle={{ color: '#94a3b8' }}
            formatter={(v) => [v != null ? `${v}%` : '']}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
          {METRICS.map(m => (
            <Bar key={m.label} dataKey={m.label} fill={m.color} radius={[3, 3, 0, 0]} maxBarSize={32} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
