'use client'
import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, AlertCircle, Bot, Cpu } from 'lucide-react'
import type { AskResponse, AgentResponse } from '@/types'
import { askQuestion, agentAsk } from '@/lib/api'
import { SourceCard } from './SourceCard'
import { CRAGTrace } from './CRAGTrace'
import { AgentTrace } from './AgentTrace'

type AnyResponse = AskResponse | AgentResponse

interface Message {
  id: string
  question: string
  response: AnyResponse | null
  error: string | null
  loading: boolean
  mode: 'standard' | 'agent'
}

const EXAMPLES = [
  'What machine learning model is used for anomaly detection?',
  'How does data flow from a CSV file to a prediction?',
  'What are the main sources of technical debt in this codebase?',
  'Compare the fit and predict methods of AnomalyDetector',
]

const AGENT_EXAMPLES = [
  'Generate a Mermaid diagram of AnomalyDetector and its methods',
  'Suggest refactors for the fit function',
  'Run: print("Hello from the codebase!")',
  'Search for all anomaly detection classes and explain how they relate',
]

function isAgentResponse(r: AnyResponse): r is AgentResponse {
  return 'tool_calls' in r
}

interface Props {
  onResponse: (r: AnyResponse) => void
}

export function ChatPanel({ onResponse }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState<'standard' | 'agent'>('standard')
  const [sessionId] = useState(() => crypto.randomUUID())
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function submit(question: string) {
    if (!question.trim() || loading) return
    const id = Date.now().toString()
    setInput('')
    setLoading(true)

    setMessages(prev => [...prev, { id, question, response: null, error: null, loading: true, mode }])

    try {
      const res = mode === 'agent'
        ? await agentAsk(question, sessionId)
        : await askQuestion(question)
      setMessages(prev =>
        prev.map(m => m.id === id ? { ...m, response: res, loading: false } : m)
      )
      onResponse(res)
    } catch (e: any) {
      setMessages(prev =>
        prev.map(m => m.id === id ? { ...m, error: e.message, loading: false } : m)
      )
    } finally {
      setLoading(false)
    }
  }

  const examples = mode === 'agent' ? AGENT_EXAMPLES : EXAMPLES

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle */}
      <div className="flex gap-1 px-4 pt-3 pb-1 shrink-0">
        <button
          onClick={() => setMode('standard')}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${
            mode === 'standard'
              ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300'
              : 'border-[#2a2d3e] text-[#64748b] hover:text-[#94a3b8]'
          }`}
        >
          <Cpu className="w-3 h-3" />
          Standard
        </button>
        <button
          onClick={() => setMode('agent')}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${
            mode === 'agent'
              ? 'bg-emerald-600/20 border-emerald-500/40 text-emerald-300'
              : 'border-[#2a2d3e] text-[#64748b] hover:text-[#94a3b8]'
          }`}
        >
          <Bot className="w-3 h-3" />
          Agentic
        </button>
        {mode === 'agent' && (
          <span className="ml-auto text-[10px] text-[#475569] self-center">memory: on</span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div>
              <h2 className="text-xl font-semibold text-[#e2e8f0] mb-2">Code-RAG</h2>
              <p className="text-[#64748b] text-sm max-w-xs">
                {mode === 'agent'
                  ? 'Agentic mode: uses tools to search, run code, generate diagrams, and suggest refactors.'
                  : 'Ask anything about the codebase. Routes to simple, complex, or conceptual paths automatically.'}
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
              {examples.map(ex => (
                <button
                  key={ex}
                  onClick={() => submit(ex)}
                  className="text-left text-xs text-[#94a3b8] bg-[#1a1d27] border border-[#2a2d3e] rounded-lg px-3 py-2 hover:border-indigo-500/50 hover:text-[#e2e8f0] transition-all"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className="space-y-3">
            {/* Question */}
            <div className="flex justify-end">
              <div className={`border rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-[#e2e8f0] max-w-[85%] ${
                msg.mode === 'agent'
                  ? 'bg-emerald-600/10 border-emerald-500/20'
                  : 'bg-indigo-600/20 border-indigo-500/30'
              }`}>
                {msg.question}
              </div>
            </div>

            {/* Loading */}
            {msg.loading && (
              <div className="flex items-center gap-2 text-[#64748b] text-sm pl-1">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>{msg.mode === 'agent' ? 'Agent thinking...' : 'Thinking...'}</span>
              </div>
            )}

            {/* Error */}
            {msg.error && (
              <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {msg.error}
              </div>
            )}

            {/* Response */}
            {msg.response && (
              <div className="space-y-3">
                {isAgentResponse(msg.response) ? (
                  <AgentTrace
                    toolCalls={msg.response.tool_calls}
                    sessionId={msg.response.session_id}
                  />
                ) : (
                  <CRAGTrace trace={msg.response.trace} />
                )}

                <div className="text-sm text-[#e2e8f0] leading-relaxed bg-[#1a1d27] border border-[#2a2d3e] rounded-lg px-4 py-3 whitespace-pre-wrap">
                  {msg.response.answer}
                </div>

                {msg.response.sources.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs text-[#64748b] px-1">Sources</p>
                    {msg.response.sources.slice(0, 6).map((src, i) => (
                      <SourceCard key={src.node_id} node={src} index={i + 1} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-[#2a2d3e]">
        <form
          onSubmit={e => { e.preventDefault(); submit(input) }}
          className="flex gap-2"
        >
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={mode === 'agent' ? 'Ask, run code, or request a diagram...' : 'Ask about the codebase...'}
            disabled={loading}
            className="flex-1 bg-[#1a1d27] border border-[#2a2d3e] rounded-lg px-4 py-2.5 text-sm text-[#e2e8f0] placeholder-[#64748b] focus:outline-none focus:border-indigo-500/60 disabled:opacity-50 transition-colors"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className={`disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg px-4 py-2.5 transition-colors flex items-center gap-1.5 text-sm font-medium ${
              mode === 'agent'
                ? 'bg-emerald-600 hover:bg-emerald-500'
                : 'bg-indigo-600 hover:bg-indigo-500'
            }`}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </form>
      </div>
    </div>
  )
}
