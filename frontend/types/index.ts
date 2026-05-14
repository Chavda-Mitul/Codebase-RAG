export type NodeType = 'File' | 'Class' | 'Function' | 'Concept' | 'Module'
export type RouteType = 'simple' | 'complex' | 'conceptual'

export interface SourceNode {
  node_id: string
  label: NodeType
  name: string
  score: number
  source: string
}

export interface DocGrade {
  node_id: string
  name: string
  score: 'yes' | 'no'
  reason: string
}

export interface TraceInfo {
  route: RouteType
  iterations: number
  query_rewrites: string[]
  correction_triggered: boolean
  hallucination_check: string
  answer_check: string
  doc_grades: DocGrade[]
  step_back_question: string
  sub_questions: string[]
}

export interface GraphNodeData {
  name: string
  nodeType: NodeType
  isSource: boolean
  docstring?: string
  file_path?: string
}

export interface FlowNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: GraphNodeData
}

export interface FlowEdge {
  id: string
  source: string
  target: string
  label: string
  animated?: boolean
}

export interface GraphData {
  nodes: FlowNode[]
  edges: FlowEdge[]
}

export interface AskResponse {
  answer: string
  question: string
  trace: TraceInfo
  sources: SourceNode[]
  graph: GraphData
}

export interface GraphStatsResponse {
  node_counts: Record<string, number>
  relationship_counts: Record<string, number>
  total_nodes: number
  total_relationships: number
}

export interface EvalScores {
  faithfulness: number
  answer_relevance: number
  context_recall: number
  context_precision: number
  avg_overall?: number
}

export interface EvalQuestion {
  id: number
  run_id: number
  qa_id: string
  question: string
  answer: string
  faithfulness: number
  answer_relevance: number
  context_recall: number
  context_precision: number
  latency_ms: number
}

export interface EvalRun {
  id: number
  pipeline: string
  timestamp: string
  question_count: number
  avg_faithfulness: number
  avg_answer_relevance: number
  avg_context_recall: number
  avg_context_precision: number
  avg_overall: number
}

export interface EvalRunDetail extends EvalRun {
  questions: EvalQuestion[]
}

export type PipelineComparison = Record<string, EvalRun>

export interface ToolCallInfo {
  tool: string
  input: string
  output: string
}

export interface AgentResponse {
  answer: string
  question: string
  tool_calls: ToolCallInfo[]
  sources: SourceNode[]
  graph: GraphData
  session_id: string
}
