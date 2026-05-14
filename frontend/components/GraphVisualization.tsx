'use client'
import { useCallback } from 'react'
import {
  ReactFlow, Background, Controls, MiniMap,
  BackgroundVariant, type NodeTypes, type Node, type Edge,
  Handle, Position,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { GraphData, GraphNodeData } from '@/types'
import clsx from 'clsx'

const NODE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  File:     { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd' },
  Class:    { bg: '#3b1f6e', border: '#8b5cf6', text: '#c4b5fd' },
  Function: { bg: '#14432a', border: '#10b981', text: '#6ee7b7' },
  Concept:  { bg: '#452a0a', border: '#f59e0b', text: '#fcd34d' },
  Module:   { bg: '#1e293b', border: '#64748b', text: '#94a3b8' },
  Node:     { bg: '#1e293b', border: '#64748b', text: '#94a3b8' },
}

function CodeNode({ data }: { data: GraphNodeData }) {
  const colors = NODE_COLORS[data.nodeType] || NODE_COLORS.Node
  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs min-w-[140px] max-w-[200px] shadow-lg"
      style={{ background: colors.bg, borderColor: colors.border }}
    >
      <Handle type="target" position={Position.Left} style={{ background: colors.border, width: 8, height: 8 }} />
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: colors.border }}>
          {data.nodeType}
        </span>
        {data.isSource && (
          <span className="text-[9px] bg-indigo-500/30 text-indigo-300 rounded px-1 ml-auto">source</span>
        )}
      </div>
      <div className="font-mono font-medium truncate" style={{ color: colors.text }} title={data.name}>
        {data.name}
      </div>
      {data.docstring && (
        <div className="text-[#64748b] mt-1 text-[10px] line-clamp-2 leading-tight">
          {data.docstring}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: colors.border, width: 8, height: 8 }} />
    </div>
  )
}

const nodeTypes: NodeTypes = { codeNode: CodeNode }

interface Props {
  graph: GraphData
  stats?: { total_nodes: number; total_relationships: number }
}

export function GraphVisualization({ graph, stats }: Props) {
  const nodes: Node[] = graph.nodes.map(n => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: n.data,
  }))

  const edges: Edge[] = graph.edges.map(e => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label,
    animated: e.animated,
    style: { stroke: '#374151', strokeWidth: 1.5 },
    labelStyle: { fill: '#64748b', fontSize: 9 },
    labelBgStyle: { fill: '#1a1d27' },
  }))

  return (
    <div className="relative h-full w-full rounded-lg overflow-hidden border border-[#2a2d3e]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#2a2d3e" />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={n => NODE_COLORS[(n.data as GraphNodeData)?.nodeType]?.border || '#64748b'}
          maskColor="rgba(15,17,23,0.7)"
          style={{ background: '#1a1d27' }}
        />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-12 left-3 flex flex-col gap-1 bg-[#1a1d27]/90 rounded-lg border border-[#2a2d3e] p-2">
        {Object.entries(NODE_COLORS).filter(([k]) => k !== 'Node').map(([type, c]) => (
          <div key={type} className="flex items-center gap-1.5 text-[10px]">
            <div className="w-2 h-2 rounded-sm border" style={{ background: c.bg, borderColor: c.border }} />
            <span style={{ color: c.text }}>{type}</span>
          </div>
        ))}
      </div>

      {/* Stats overlay */}
      {stats && (
        <div className="absolute top-3 right-3 bg-[#1a1d27]/90 rounded-lg border border-[#2a2d3e] px-3 py-1.5 text-xs text-[#64748b]">
          <span className="text-[#94a3b8]">{stats.total_nodes}</span> nodes ·{' '}
          <span className="text-[#94a3b8]">{stats.total_relationships}</span> rels
        </div>
      )}

      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-[#64748b] text-sm">
          Ask a question to see the knowledge graph
        </div>
      )}
    </div>
  )
}
