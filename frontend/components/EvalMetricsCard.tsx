'use client'

interface Props {
  metric: string
  label: string
  score: number
  description: string
}

function scoreColor(s: number): string {
  if (s >= 0.8) return '#10b981'   // green
  if (s >= 0.6) return '#f59e0b'   // amber
  return '#ef4444'                  // red
}

function scoreBg(s: number): string {
  if (s >= 0.8) return 'bg-emerald-500/10 border-emerald-500/20'
  if (s >= 0.6) return 'bg-amber-500/10 border-amber-500/20'
  return 'bg-red-500/10 border-red-500/20'
}

export function EvalMetricsCard({ metric, label, score, description }: Props) {
  const pct = Math.round(score * 100)
  const color = scoreColor(score)
  const bg = scoreBg(score)

  return (
    <div className={`rounded-xl border p-4 flex flex-col gap-3 ${bg}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-[#94a3b8] uppercase tracking-wide">{label}</span>
        <span className="text-2xl font-bold" style={{ color }}>{pct}%</span>
      </div>

      {/* Bar */}
      <div className="h-1.5 bg-[#1a1d27] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>

      <p className="text-xs text-[#64748b] leading-snug">{description}</p>
    </div>
  )
}
