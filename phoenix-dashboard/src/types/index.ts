// ============================================================
// Phoenix Dashboard – Domain types
// Derived from the SSE stream + proxy log formats
// ============================================================

/** Pipeline phases in execution order */
export const PIPELINE_PHASES = [
  'comprehend',
  'retrieve',
  'tree_research',
  'entities',
  'verify',
  'reflect',
  'persist',
  'synthesize',
] as const;

export type PipelinePhase = (typeof PIPELINE_PHASES)[number];

export type PhaseStatus = 'pending' | 'active' | 'completed' | 'error';

/** Mapped labels for display */
export const PHASE_LABELS: Record<PipelinePhase, string> = {
  comprehend: 'Phase 0: Query Comprehension',
  retrieve: 'Phase 1: Retrieve Prior Knowledge',
  tree_research: 'Phase 2: Tree Research Reactor',
  entities: 'Phase 4: Knowledge Graph Update',
  verify: 'Phase 5: Citation Cross-Check',
  reflect: 'Phase 6: AoT Reflection',
  persist: 'Phase 7: Persist Knowledge',
  synthesize: 'Synthesis Phase',
};

/** A single pipeline phase with its state */
export interface PipelineNode {
  id: PipelinePhase;
  label: string;
  status: PhaseStatus;
  startTime?: number;
  endTime?: number;
  details: string[];
  iteration: number; // which re-research iteration (0 = first pass)
}

/** An edge between pipeline phases */
export interface PipelineEdge {
  source: PipelinePhase;
  target: PipelinePhase;
  label?: string;
  isConditional?: boolean;
}

// ---- Tree Research Structures ----

export interface ResearchTreeNode {
  id: string;
  question: string;
  depth: number;
  parentId: string | null;
  status: 'active' | 'completed' | 'saturated';
  findings: number;
  toolCalls: ToolCall[];
  conditions: Condition[];
  turns: number;
  totalToolCalls: number;
  childrenSpawned: number;
  novelty?: number;
  /** Key finding summary */
  keySummary?: string;
}

export interface ToolCall {
  id: string;
  tool: string;
  query: string;
  duration: number; // seconds
  timestamp: number;
  nodeId: string;
}

export interface Condition {
  id: string;
  text: string;
  confidence: 'high' | 'low' | 'speculative';
  nodeId: string;
  admitted: boolean;
}

// ---- Subagent Structures ----

export interface SubagentInfo {
  id: string; // e.g. "sa0", "sa0-d1"
  question: string;
  depth: number;
  parentSubagentId: string | null;
  status: 'active' | 'completed' | 'saturated';
  conditions: number;
  turns: number;
  toolCalls: number;
  childrenSpawned: number;
  novelty: number;
  toolCallDetails: ToolCall[];
}

// ---- Metrics ----

export interface AdmissionStats {
  admitted: number;
  duplicatesRejected: number;
  irrelevantRejected: number;
  iteration: number;
}

export interface TreeExplorationStats {
  nodesExplored: number;
  depthReached: number;
  atomicConditions: number;
  totalTurns: number;
  totalToolCalls: number;
  iteration: number;
}

export interface CrossCheckStats {
  highConfidence: number;
  lowConfidence: number;
  speculative: number;
  totalChecked: number;
  iteration: number;
}

export interface ComprehensionStats {
  entities: number;
  domains: number;
  implicitQuestions: number;
  adjacentTerritories: number;
  iteration: number;
}

export interface PersistStats {
  newConditions: number;
  skippedAlreadyPersisted: number;
  iteration: number;
}

export interface EntityStats {
  entities: number;
  relationships: number;
  newEntities: number;
  newEdges: number;
  iteration: number;
}

export interface ReflectionStats {
  decompositionQuality: number;
  iteration: number;
}

/** Aggregate metrics for the whole run */
export interface RunMetrics {
  comprehension: ComprehensionStats[];
  treeExploration: TreeExplorationStats[];
  admission: AdmissionStats[];
  crossCheck: CrossCheckStats[];
  persist: PersistStats[];
  entities: EntityStats[];
  reflection: ReflectionStats[];
  toolCallsByEngine: Record<string, number>;
  totalIterations: number;
}

// ---- SSE Event ----

export interface SSEEvent {
  id: string;
  content: string;
  timestamp: number;
  lineNumber: number;
}

// ---- Log Event ----

export interface LogEvent {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  agentId: string; // e.g. "req-a55ca693-sa0-d1"
  message: string;
  lineNumber: number;
}

// ---- Drilldown navigation ----

export type DrilldownLevel = 'pipeline' | 'tree' | 'subagent' | 'condition';

export interface DrilldownState {
  level: DrilldownLevel;
  /** Selected pipeline phase (for tree drilldown) */
  selectedPhase?: PipelinePhase;
  /** Selected tree node (for subagent drilldown) */
  selectedTreeNode?: string;
  /** Selected subagent (for condition drilldown) */
  selectedSubagent?: string;
  /** Current iteration */
  iteration: number;
}
