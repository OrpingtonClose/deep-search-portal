// ============================================================
// Level 0: Pipeline DAG – Top-level view of research phases
// ============================================================

import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  Position,
  MarkerType,
  Handle,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { motion } from 'framer-motion';
import { useStore } from '../../store/useStore';
import type { PipelineNode, PhaseStatus } from '../../types';

const STATUS_COLORS: Record<PhaseStatus, string> = {
  pending: '#374151',   // gray-700
  active: '#2563eb',    // blue-600
  completed: '#16a34a', // green-600
  error: '#dc2626',     // red-600
};

const STATUS_BG: Record<PhaseStatus, string> = {
  pending: '#1f2937',
  active: '#1e3a5f',
  completed: '#14532d',
  error: '#450a0a',
};

/** Custom node component for pipeline phases */
function PipelinePhaseNode({ data }: { data: PipelineNode & { onDrilldown: () => void } }) {
  const isActive = data.status === 'active';
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="relative cursor-pointer select-none"
      style={{
        background: STATUS_BG[data.status],
        border: `2px solid ${STATUS_COLORS[data.status]}`,
        borderRadius: 12,
        padding: '12px 20px',
        minWidth: 180,
        textAlign: 'center',
        boxShadow: isActive
          ? `0 0 20px ${STATUS_COLORS[data.status]}40`
          : '0 2px 8px rgba(0,0,0,0.3)',
      }}
      onClick={data.onDrilldown}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
      <div className="text-xs font-medium text-gray-400 mb-1">{data.label}</div>
      <div className="text-sm font-bold text-white capitalize">{data.id.replace('_', ' ')}</div>
      {isActive && (
        <motion.div
          className="absolute -top-1 -right-1 w-3 h-3 rounded-full"
          style={{ background: STATUS_COLORS.active }}
          animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        />
      )}
      {data.iteration > 0 && (
        <div className="absolute -top-2 -left-2 bg-amber-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
          iter {data.iteration}
        </div>
      )}
      {data.details.length > 0 && (
        <div className="text-[10px] text-gray-500 mt-1 truncate max-w-[160px]">
          {data.details[data.details.length - 1]?.slice(0, 60)}...
        </div>
      )}
    </motion.div>
  );
}

const nodeTypes = { pipelinePhase: PipelinePhaseNode };

export function PipelineDAG() {
  const pipelineNodes = useStore(s => s.pipelineNodes);
  const pipelineEdges = useStore(s => s.pipelineEdges);
  const setDrilldown = useStore(s => s.setDrilldown);

  const handleDrilldown = useCallback(
    (phaseId: string) => {
      if (phaseId === 'tree_research') {
        setDrilldown({ level: 'tree', selectedPhase: 'tree_research' });
      }
    },
    [setDrilldown]
  );

  const nodes: Node[] = useMemo(() => {
    // Layout: 2 columns for the main path, conditional edges loop back
    const positions: Record<string, { x: number; y: number }> = {
      comprehend: { x: 300, y: 0 },
      retrieve: { x: 300, y: 100 },
      tree_research: { x: 300, y: 200 },
      entities: { x: 300, y: 300 },
      verify: { x: 300, y: 400 },
      reflect: { x: 300, y: 500 },
      persist: { x: 300, y: 620 },
      synthesize: { x: 300, y: 740 },
    };

    return pipelineNodes.map(node => ({
      id: node.id,
      type: 'pipelinePhase',
      position: positions[node.id] || { x: 0, y: 0 },
      data: { ...node, onDrilldown: () => handleDrilldown(node.id) },
    }));
  }, [pipelineNodes, handleDrilldown]);

  const edges: Edge[] = useMemo(() => {
    return pipelineEdges.map((e, i) => ({
      id: `edge-${i}`,
      source: e.source,
      target: e.target,
      label: e.label,
      type: 'smoothstep',
      animated: e.isConditional,
      style: {
        stroke: e.isConditional ? '#f59e0b' : '#6b7280',
        strokeWidth: 2,
      },
      labelStyle: { fill: '#9ca3af', fontSize: 10 },
      markerEnd: { type: MarkerType.ArrowClosed, color: e.isConditional ? '#f59e0b' : '#6b7280' },
    }));
  }, [pipelineEdges]);

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={2}
      >
        <Background color="#374151" gap={20} />
        <Controls
          style={{ background: '#1f2937', borderColor: '#374151' }}
        />
      </ReactFlow>
    </div>
  );
}
