// ============================================================
// SSE Stream Parser
// Parses the OpenAI-compatible SSE stream from the persistent proxy
// into structured graph data for visualization
// ============================================================

import type {
  PipelineNode,
  PipelineEdge,
  PipelinePhase,
  ResearchTreeNode,
  ToolCall,
  RunMetrics,
  SSEEvent,
} from '../types';
import { PHASE_LABELS } from '../types';

/** Extract content string from an SSE data line */
function extractContent(line: string): string | null {
  if (!line.startsWith('data: ')) return null;
  try {
    const json = JSON.parse(line.slice(6));
    const content = json?.choices?.[0]?.delta?.content;
    return typeof content === 'string' ? content : null;
  } catch {
    return null;
  }
}

/** Phase detection patterns - maps regex to pipeline phase */
const PHASE_PATTERNS: [RegExp, PipelinePhase][] = [
  [/\*\*\[Phase 0: Query Comprehension\]\*\*/, 'comprehend'],
  [/\*\*\[Phase 1: Retrieving Prior Knowledge\]\*\*/, 'retrieve'],
  [/\*\*\[Phase 2: Tree Research Reactor\]\*\*/, 'tree_research'],
  [/\*\*\[Phase 2a: Query Comprehension\]\*\*/, 'comprehend'], // re-research
  [/\*\*\[Phase 4: Knowledge Graph Update\]\*\*/, 'entities'],
  [/\*\*\[Phase 5: Citation Cross-Check\]\*\*/, 'verify'],
  [/\*\*\[Phase 5b: Inline Verification\]\*\*/, 'verify'],
  [/\*\*\[Phase 6: AoT Reflection\]\*\*/, 'reflect'],
  [/\*\*\[Phase 7: Persisting Knowledge\]\*\*/, 'persist'],
  [/\*\*\[Synthesis Phase\]\*\*/, 'synthesize'],
  [/\*\*\[Re-research iteration/, 'tree_research'],
];

/** Tool name extraction from query lines */
const TOOL_PATTERN = /^Querying ([\w\s/]+(?:\(.*?\))?): ["""](.+?)["""] \((\d+\.?\d*)s\)/;
const READING_PATTERN = /^Reading ([\w.]+) \((\d+\.?\d*)s\)/;

/** Tree research patterns */
const TREE_COMPLETE_PATTERN = /\*\*Tree Exploration Complete:\*\* (\d+) nodes explored \(depth reached: (\d+)\), (\d+) atomic conditions, (\d+) total turns, (\d+) tool calls/;
const ADMISSION_PATTERN = /\*\*Admission Pipeline Stats:\*\* (\d+) admitted, (\d+) duplicates rejected, (\d+) irrelevant rejected/;
const CROSSCHECK_PATTERN = /Cross-check complete: (\d+) high-confidence, (\d+) low-confidence, (\d+) speculative/;
const CROSSCHECK_START_PATTERN = /Cross-checking (\d+) pre-admitted conditions/;

/** Comprehension patterns */
const COMPREHENSION_PATTERN = /Identified (\d+) entities, (\d+) domains, (\d+) implicit questions, (\d+) adjacent territories/;

/** Entity extraction patterns */
const ENTITY_PATTERN = /Extracted (\d+) entities, (\d+) relationships\. Stored (\d+) new entities, (\d+) new edges/;

/** Persist patterns */
const PERSIST_PATTERN = /Persisting (\d+) new conditions \(skipping (\d+) already-persisted\)/;
const STORED_PATTERN = /Stored (\d+) conditions to persistent knowledge base/;

/** Reflection patterns */
const REFLECTION_PATTERN = /Decomposition quality: ([\d.]+)\/([\d.]+)/;

/** Tree node patterns from SSE content */
const INVESTIGATING_PATTERN = /^Investigating: (.+)/;
const QUESTION_FINDINGS_PATTERN = /^(?:\[depth (\d+)\] )?(.+?) — (\d+) findings\. Key: (.+)/;
const SPAWNING_PATTERN = /^Spawning (\d+) follow-up questions? — highest priority: (.+)/;
const RERESEARCH_PATTERN = /\*\*\[Re-research iteration (\d+)\]\*\* Targeting (\d+) gap questions/;

/** Parse the full SSE text into structured data */
export function parseSSEStream(text: string): ParsedStream {
  const lines = text.split('\n');
  const events: SSEEvent[] = [];
  const pipelineNodes = createInitialPipeline();
  const pipelineEdges = createPipelineEdges();
  const treeNodes: Map<string, ResearchTreeNode> = new Map();
  const toolCalls: ToolCall[] = [];
  const metrics = createEmptyMetrics();

  let currentPhase: PipelinePhase | null = null;
  let currentIteration = 0;
  let treeNodeCounter = 0;
  let toolCallCounter = 0;
  let currentInvestigation: string | null = null;
  let lineIdx = 0;

  for (const line of lines) {
    lineIdx++;
    const content = extractContent(line);
    if (!content) continue;

    const trimmed = content.trim();
    if (!trimmed) continue;

    events.push({
      id: `evt-${lineIdx}`,
      content: trimmed,
      timestamp: lineIdx, // relative ordering
      lineNumber: lineIdx,
    });

    // --- Phase detection ---
    for (const [pattern, phase] of PHASE_PATTERNS) {
      if (pattern.test(trimmed)) {
        // Complete previous phase
        if (currentPhase) {
          const prevNode = pipelineNodes.find(n => n.id === currentPhase);
          if (prevNode && prevNode.status === 'active') {
            prevNode.status = 'completed';
            prevNode.endTime = lineIdx;
          }
        }

        // Check for re-research
        const reresearchMatch = trimmed.match(RERESEARCH_PATTERN);
        if (reresearchMatch) {
          currentIteration = parseInt(reresearchMatch[1], 10);
          metrics.totalIterations = Math.max(metrics.totalIterations, currentIteration);
          // Reset phases for new iteration
          for (const node of pipelineNodes) {
            if (['tree_research', 'entities', 'verify', 'reflect', 'persist', 'synthesize'].includes(node.id)) {
              node.status = 'pending';
              node.iteration = currentIteration;
            }
          }
        }

        const targetNode = pipelineNodes.find(n => n.id === phase);
        if (targetNode) {
          targetNode.status = 'active';
          targetNode.startTime = lineIdx;
          targetNode.iteration = currentIteration;
          targetNode.details.push(trimmed);
        }
        currentPhase = phase;
        break;
      }
    }

    // --- Add details to current phase ---
    if (currentPhase) {
      const node = pipelineNodes.find(n => n.id === currentPhase);
      if (node && !PHASE_PATTERNS.some(([p]) => p.test(trimmed))) {
        // Don't duplicate phase headers as details
        if (node.details.length < 50) {
          node.details.push(trimmed);
        }
      }
    }

    // --- Tool call extraction ---
    const toolMatch = trimmed.match(TOOL_PATTERN);
    if (toolMatch) {
      const tc: ToolCall = {
        id: `tc-${toolCallCounter++}`,
        tool: toolMatch[1].trim(),
        query: toolMatch[2],
        duration: parseFloat(toolMatch[3]),
        timestamp: lineIdx,
        nodeId: currentInvestigation || 'unknown',
      };
      toolCalls.push(tc);
      // Count by engine
      const engine = tc.tool.split(' ')[0];
      metrics.toolCallsByEngine[engine] = (metrics.toolCallsByEngine[engine] || 0) + 1;
    }

    const readMatch = trimmed.match(READING_PATTERN);
    if (readMatch) {
      const tc: ToolCall = {
        id: `tc-${toolCallCounter++}`,
        tool: 'web_read',
        query: readMatch[1],
        duration: parseFloat(readMatch[2]),
        timestamp: lineIdx,
        nodeId: currentInvestigation || 'unknown',
      };
      toolCalls.push(tc);
      metrics.toolCallsByEngine['web_read'] = (metrics.toolCallsByEngine['web_read'] || 0) + 1;
    }

    // --- Tree node extraction ---
    const investigatingMatch = trimmed.match(INVESTIGATING_PATTERN);
    if (investigatingMatch) {
      const question = investigatingMatch[1];
      const nodeId = `tree-${treeNodeCounter++}`;
      currentInvestigation = nodeId;
      if (!treeNodes.has(nodeId)) {
        treeNodes.set(nodeId, {
          id: nodeId,
          question,
          depth: 0,
          parentId: null,
          status: 'active',
          findings: 0,
          toolCalls: [],
          conditions: [],
          turns: 0,
          totalToolCalls: 0,
          childrenSpawned: 0,
        });
      }
    }

    // Question with findings
    const findingsMatch = trimmed.match(QUESTION_FINDINGS_PATTERN);
    if (findingsMatch) {
      const depth = findingsMatch[1] ? parseInt(findingsMatch[1], 10) : 0;
      const question = findingsMatch[2];
      const findings = parseInt(findingsMatch[3], 10);
      const key = findingsMatch[4];
      const nodeId = `tree-${treeNodeCounter++}`;
      treeNodes.set(nodeId, {
        id: nodeId,
        question,
        depth,
        parentId: currentInvestigation,
        status: 'completed',
        findings,
        toolCalls: [],
        conditions: [],
        turns: 0,
        totalToolCalls: 0,
        childrenSpawned: 0,
        keySummary: key,
      });
    }

    // Spawning follow-ups
    const spawnMatch = trimmed.match(SPAWNING_PATTERN);
    if (spawnMatch) {
      const count = parseInt(spawnMatch[1], 10);
      if (currentInvestigation) {
        const node = findNodeByIdOrLast(treeNodes);
        if (node) {
          node.childrenSpawned += count;
        }
      }
    }

    // --- Metric extraction ---
    const compMatch = trimmed.match(COMPREHENSION_PATTERN);
    if (compMatch) {
      metrics.comprehension.push({
        entities: parseInt(compMatch[1], 10),
        domains: parseInt(compMatch[2], 10),
        implicitQuestions: parseInt(compMatch[3], 10),
        adjacentTerritories: parseInt(compMatch[4], 10),
        iteration: currentIteration,
      });
    }

    const treeCompleteMatch = trimmed.match(TREE_COMPLETE_PATTERN);
    if (treeCompleteMatch) {
      metrics.treeExploration.push({
        nodesExplored: parseInt(treeCompleteMatch[1], 10),
        depthReached: parseInt(treeCompleteMatch[2], 10),
        atomicConditions: parseInt(treeCompleteMatch[3], 10),
        totalTurns: parseInt(treeCompleteMatch[4], 10),
        totalToolCalls: parseInt(treeCompleteMatch[5], 10),
        iteration: currentIteration,
      });
    }

    const admissionMatch = trimmed.match(ADMISSION_PATTERN);
    if (admissionMatch) {
      metrics.admission.push({
        admitted: parseInt(admissionMatch[1], 10),
        duplicatesRejected: parseInt(admissionMatch[2], 10),
        irrelevantRejected: parseInt(admissionMatch[3], 10),
        iteration: currentIteration,
      });
    }

    const crosscheckMatch = trimmed.match(CROSSCHECK_PATTERN);
    if (crosscheckMatch) {
      const totalMatch = trimmed.match(CROSSCHECK_START_PATTERN);
      metrics.crossCheck.push({
        highConfidence: parseInt(crosscheckMatch[1], 10),
        lowConfidence: parseInt(crosscheckMatch[2], 10),
        speculative: parseInt(crosscheckMatch[3], 10),
        totalChecked: totalMatch ? parseInt(totalMatch[1], 10) : 0,
        iteration: currentIteration,
      });
    }

    const entityMatch = trimmed.match(ENTITY_PATTERN);
    if (entityMatch) {
      metrics.entities.push({
        entities: parseInt(entityMatch[1], 10),
        relationships: parseInt(entityMatch[2], 10),
        newEntities: parseInt(entityMatch[3], 10),
        newEdges: parseInt(entityMatch[4], 10),
        iteration: currentIteration,
      });
    }

    const persistMatch = trimmed.match(PERSIST_PATTERN);
    if (persistMatch) {
      metrics.persist.push({
        newConditions: parseInt(persistMatch[1], 10),
        skippedAlreadyPersisted: parseInt(persistMatch[2], 10),
        iteration: currentIteration,
      });
    }

    const storedMatch = trimmed.match(STORED_PATTERN);
    if (storedMatch && metrics.persist.length > 0) {
      // Update the last persist entry
    }

    const reflectionMatch = trimmed.match(REFLECTION_PATTERN);
    if (reflectionMatch) {
      metrics.reflection.push({
        decompositionQuality: parseFloat(reflectionMatch[1]),
        iteration: currentIteration,
      });
    }
  }

  // Mark last active phase as completed
  if (currentPhase) {
    const lastNode = pipelineNodes.find(n => n.id === currentPhase);
    if (lastNode && lastNode.status === 'active') {
      lastNode.status = 'completed';
      lastNode.endTime = lineIdx;
    }
  }

  return {
    events,
    pipelineNodes,
    pipelineEdges,
    treeNodes: Array.from(treeNodes.values()),
    toolCalls,
    metrics,
    totalLines: lines.length,
  };
}

function findNodeByIdOrLast(nodes: Map<string, ResearchTreeNode>): ResearchTreeNode | null {
  const arr = Array.from(nodes.values());
  return arr.length > 0 ? arr[arr.length - 1] : null;
}

function createInitialPipeline(): PipelineNode[] {
  const phases: PipelinePhase[] = [
    'comprehend', 'retrieve', 'tree_research', 'entities',
    'verify', 'reflect', 'persist', 'synthesize',
  ];
  return phases.map(id => ({
    id,
    label: PHASE_LABELS[id],
    status: 'pending',
    details: [],
    iteration: 0,
  }));
}

function createPipelineEdges(): PipelineEdge[] {
  return [
    { source: 'comprehend', target: 'retrieve' },
    { source: 'retrieve', target: 'tree_research' },
    { source: 'tree_research', target: 'entities' },
    { source: 'entities', target: 'verify' },
    { source: 'verify', target: 'reflect' },
    { source: 'reflect', target: 'persist', label: 'quality >= 0.4' },
    { source: 'reflect', target: 'tree_research', label: 'quality < 0.4', isConditional: true },
    { source: 'persist', target: 'synthesize' },
    { source: 'synthesize', target: 'tree_research', label: 'incomplete', isConditional: true },
  ];
}

function createEmptyMetrics(): RunMetrics {
  return {
    comprehension: [],
    treeExploration: [],
    admission: [],
    crossCheck: [],
    persist: [],
    entities: [],
    reflection: [],
    toolCallsByEngine: {},
    totalIterations: 0,
  };
}

export interface ParsedStream {
  events: SSEEvent[];
  pipelineNodes: PipelineNode[];
  pipelineEdges: PipelineEdge[];
  treeNodes: ResearchTreeNode[];
  toolCalls: ToolCall[];
  metrics: RunMetrics;
  totalLines: number;
}
