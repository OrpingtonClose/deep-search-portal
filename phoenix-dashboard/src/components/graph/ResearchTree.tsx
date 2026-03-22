// ============================================================
// Level 1: Research Tree – Tree reactor node visualization
// ============================================================

import { useMemo, useCallback } from 'react';
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
import type { ResearchTreeNode } from '../../types';

const DEPTH_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

function TreeNodeComponent({ data }: { data: ResearchTreeNode & { onClick: () => void } }) {
  const depthColor = DEPTH_COLORS[data.depth % DEPTH_COLORS.length];
  const isSaturated = data.status === 'saturated';

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="cursor-pointer select-none"
      onClick={data.onClick}
      style={{
        background: '#1a1a2e',
        border: `2px solid ${isSaturated ? '#6b7280' : depthColor}`,
        borderRadius: 10,
        padding: '10px 14px',
        minWidth: 200,
        maxWidth: 300,
        opacity: isSaturated ? 0.6 : 1,
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: depthColor }} />
      <Handle type="source" position={Position.Bottom} style={{ background: depthColor }} />
      <div className="flex items-center gap-2 mb-1">
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ background: depthColor }}
        />
        <span className="text-[10px] font-mono text-gray-500">
          depth {data.depth}
        </span>
        {data.findings > 0 && (
          <span className="text-[10px] text-emerald-400 ml-auto">
            {data.findings} findings
          </span>
        )}
      </div>
      <div className="text-xs text-gray-200 line-clamp-2 leading-tight">
        {data.question.slice(0, 120)}{data.question.length > 120 ? '...' : ''}
      </div>
      {data.keySummary && (
        <div className="text-[10px] text-amber-300/70 mt-1 line-clamp-1">
          Key: {data.keySummary.slice(0, 80)}...
        </div>
      )}
      {data.childrenSpawned > 0 && (
        <div className="text-[10px] text-purple-400 mt-1">
          +{data.childrenSpawned} follow-ups
        </div>
      )}
    </motion.div>
  );
}

const nodeTypes = { treeNode: TreeNodeComponent };

export function ResearchTree() {
  const treeNodes = useStore(s => s.treeNodes);
  const setDrilldown = useStore(s => s.setDrilldown);

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      setDrilldown({ level: 'subagent', selectedTreeNode: nodeId });
    },
    [setDrilldown]
  );

  const { nodes, edges } = useMemo(() => {
    // Group nodes by depth, then lay out in a tree structure
    const depthGroups: Map<number, ResearchTreeNode[]> = new Map();
    for (const tn of treeNodes) {
      const group = depthGroups.get(tn.depth) || [];
      group.push(tn);
      depthGroups.set(tn.depth, group);
    }

    const flowNodes: Node[] = [];
    const flowEdges: Edge[] = [];
    const Y_SPACING = 140;
    const X_SPACING = 320;

    for (const [depth, group] of depthGroups) {
      const totalWidth = group.length * X_SPACING;
      const startX = -totalWidth / 2;

      group.forEach((tn, i) => {
        flowNodes.push({
          id: tn.id,
          type: 'treeNode',
          position: { x: startX + i * X_SPACING, y: depth * Y_SPACING },
          data: { ...tn, onClick: () => handleNodeClick(tn.id) },
        });

        if (tn.parentId) {
          flowEdges.push({
            id: `edge-${tn.id}`,
            source: tn.parentId,
            target: tn.id,
            type: 'smoothstep',
            style: { stroke: DEPTH_COLORS[tn.depth % DEPTH_COLORS.length], strokeWidth: 1.5 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: DEPTH_COLORS[tn.depth % DEPTH_COLORS.length],
            },
          });
        }
      });
    }

    return { nodes: flowNodes, edges: flowEdges };
  }, [treeNodes, handleNodeClick]);

  if (treeNodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No tree research data available. Click on the Tree Research phase to view.
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        minZoom={0.1}
        maxZoom={2}
      >
        <Background color="#374151" gap={20} />
        <Controls style={{ background: '#1f2937', borderColor: '#374151' }} />
      </ReactFlow>
    </div>
  );
}
