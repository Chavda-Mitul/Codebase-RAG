'use client'
import { useState, useEffect, useRef } from 'react'
import { ChevronDown, ChevronRight, Terminal, GitBranch, Wrench, Search, Code2 } from 'lucide-react'
import type { ToolCallInfo } from '@/types'

interface Props {
  toolCalls: ToolCallInfo[]
  sessionId: string
}

const TOOL_ICONS: Record<string, React.ReactNode> = {
  search_codebase: <Search className="w-3.5 h-3.5" />,
  run_code_snippet: <Terminal className="w-3.5 h-3.5" />,
  generate_mermaid_diagram: <GitBranch className="w-3.5 h-3.5" />,
  suggest_refactors: <Wrench className="w-3.5 h-3.5" />,
}

const TOOL_COLORS: Record<string, string> = {
  search_codebase: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  run_code_snippet: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  generate_mermaid_diagram: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  suggest_refactors: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
}

function MermaidDiagram({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    import('mermaid').then(m => {
      if (cancelled || !ref.current) return
      const id = `mermaid-${Math.random().toString(36).slice(2)}`
      m.default.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' })
      m.default.render(id, chart).then(({ svg }) => {
        if (!cancelled && ref.current) ref.current.innerHTML = svg
      }).catch(e => {
        if (!cancelled) setError(String(e))
      })
    }).catch(e => setError(String(e)))
    return () => { cancelled = true }
  }, [chart])

  if (error) return <pre className="text-red-400 text-xs">{chart}</pre>
  return <div ref={ref} className="overflow-x-auto py-2" />
}

function ToolCallCard({ tc, index }: { tc: ToolCallInfo; index: number }) {
  const [open, setOpen] = useState(index === 0)
  const color = TOOL_COLORS[tc.tool] || 'text-[#94a3b8] bg-[#1a1d27] border-[#2a2d3e]'
  const icon = TOOL_ICONS[tc.tool] || <Code2 className="w-3.5 h-3.5" />
  const isMermaid = tc.tool === 'generate_mermaid_diagram' && tc.output.trim().startsWith('graph')

  return (
    <div className="border border-[#2a2d3e] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-[#1a1d27] transition-colors"
      >
        <span className={`flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded border ${color}`}>
          {icon}
          {tc.tool.replace(/_/g, ' ')}
        </span>
        <span className="text-xs text-[#64748b] truncate flex-1">{tc.input}</span>
        {open ? <ChevronDown className="w-3.5 h-3.5 text-[#64748b] shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 text-[#64748b] shrink-0" />}
      </button>

      {open && (
        <div className="border-t border-[#2a2d3e] px-3 py-2 bg-[#0f1117]">
          {isMermaid ? (
            <MermaidDiagram chart={tc.output} />
          ) : (
            <pre className="text-xs text-[#94a3b8] whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
              {tc.output}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

export function AgentTrace({ toolCalls, sessionId }: Props) {
  const [open, setOpen] = useState(true)

  if (!toolCalls.length) return null

  return (
    <div className="border border-[#2a2d3e] rounded-lg overflow-hidden text-xs">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-[#1a1d27] hover:bg-[#1e2230] transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3 text-[#64748b]" /> : <ChevronRight className="w-3 h-3 text-[#64748b]" />}
        <span className="text-[#94a3b8] font-medium">Agent Tools</span>
        <span className="bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 rounded text-[10px]">
          {toolCalls.length} call{toolCalls.length !== 1 ? 's' : ''}
        </span>
        <span className="ml-auto text-[#475569] text-[10px]">session: {sessionId.slice(0, 8) || 'anon'}</span>
      </button>

      {open && (
        <div className="px-3 py-2 space-y-2 bg-[#0d0f17]">
          {toolCalls.map((tc, i) => (
            <ToolCallCard key={i} tc={tc} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}
