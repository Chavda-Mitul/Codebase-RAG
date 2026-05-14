'use client'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Activity, BarChart3 } from 'lucide-react'
import type { AskResponse, AgentResponse, GraphData, GraphStatsResponse } from '@/types'
import { getGraphStats } from '@/lib/api'
import { ChatPanel } from '@/components/ChatPanel'
import { GraphVisualization } from '@/components/GraphVisualization'

const EMPTY_GRAPH: GraphData = { nodes: [], edges: [] }

export default function Home() {
  const [graph, setGraph] = useState<GraphData>(EMPTY_GRAPH)
  const [stats, setStats] = useState<GraphStatsResponse | null>(null)
  const [neo4jOk, setNeo4jOk] = useState<boolean | null>(null)

  useEffect(() => {
    getGraphStats()
      .then(s => { setStats(s); setNeo4jOk(true) })
      .catch(() => setNeo4jOk(false))
  }, [])

  function handleResponse(res: AskResponse | AgentResponse) {
    setGraph(res.graph)
  }

  return (
    <div className="flex flex-col h-screen bg-[#0f1117]">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-3 border-b border-[#2a2d3e] bg-[#1a1d27] shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-indigo-500" />
          <span className="font-semibold text-[#e2e8f0] text-sm">Code-RAG</span>
          <span className="text-[#64748b] text-xs">GraphRAG + CRAG Pipeline</span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-xs">
          <Link href="/eval" className="flex items-center gap-1.5 text-[#64748b] hover:text-indigo-400 transition-colors">
            <BarChart3 className="w-3.5 h-3.5" />
            Eval
          </Link>
          {stats && (
            <span className="text-[#64748b]">
              <span className="text-[#94a3b8]">{stats.total_nodes}</span> nodes ·{' '}
              <span className="text-[#94a3b8]">{stats.total_relationships}</span> relationships
            </span>
          )}
          <div className="flex items-center gap-1.5">
            <Activity className="w-3 h-3" style={{ color: neo4jOk === true ? '#10b981' : neo4jOk === false ? '#ef4444' : '#64748b' }} />
            <span style={{ color: neo4jOk === true ? '#10b981' : neo4jOk === false ? '#ef4444' : '#64748b' }}>
              {neo4jOk === true ? 'Neo4j connected' : neo4jOk === false ? 'Neo4j offline' : 'checking...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main split layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Chat */}
        <div className="w-[480px] shrink-0 border-r border-[#2a2d3e] overflow-hidden flex flex-col">
          <ChatPanel onResponse={handleResponse} />
        </div>

        {/* Right: Graph */}
        <div className="flex-1 overflow-hidden p-3">
          <GraphVisualization
            graph={graph}
            stats={stats ?? undefined}
          />
        </div>
      </div>
    </div>
  )
}
