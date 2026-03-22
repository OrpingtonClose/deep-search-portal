// ============================================================
// Level 2: Subagent Execution View
// Shows tool calls, queries, timings for a selected tree node
// ============================================================

import { useMemo } from 'react';
import { useStore } from '../../store/useStore';
import { motion } from 'framer-motion';

const TOOL_COLORS: Record<string, string> = {
  SearXNG: '#3b82f6',
  reddit_search: '#f97316',
  forum_search: '#8b5cf6',
  telegram_search: '#06b6d4',
  hackernews_search: '#f59e0b',
  scholar_search: '#10b981',
  pubmed_search: '#ec4899',
  stackexchange_search: '#6366f1',
  wikipedia_search: '#14b8a6',
  news: '#eab308',
  Twitter: '#1d9bf0',
  web_read: '#a855f7',
  substack_search: '#f43f5e',
  b4k: '#84cc16',
  '4plebs': '#78716c',
  warosu: '#d946ef',
};

function getToolColor(tool: string): string {
  for (const [key, color] of Object.entries(TOOL_COLORS)) {
    if (tool.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return '#6b7280';
}

export function SubagentView() {
  const toolCalls = useStore(s => s.toolCalls);
  const treeNodes = useStore(s => s.treeNodes);
  const drilldown = useStore(s => s.drilldown);

  const selectedNode = useMemo(() => {
    return treeNodes.find(n => n.id === drilldown.selectedTreeNode);
  }, [treeNodes, drilldown.selectedTreeNode]);

  // Get tool calls related to this node area (by timestamp proximity)
  const relevantToolCalls = useMemo(() => {
    if (!selectedNode) return toolCalls.slice(0, 100);
    return toolCalls.filter(tc => tc.nodeId === selectedNode.id || tc.nodeId === 'unknown');
  }, [toolCalls, selectedNode]);

  // Group tool calls by tool engine
  const toolGroups = useMemo(() => {
    const groups: Record<string, typeof relevantToolCalls> = {};
    for (const tc of relevantToolCalls) {
      const engine = tc.tool.split(' ')[0];
      if (!groups[engine]) groups[engine] = [];
      groups[engine].push(tc);
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [relevantToolCalls]);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 bg-gray-900/50">
        <h3 className="text-sm font-semibold text-white mb-1">
          Subagent Execution Trace
        </h3>
        {selectedNode && (
          <p className="text-xs text-gray-400 line-clamp-2">
            {selectedNode.question}
          </p>
        )}
        <div className="flex gap-3 mt-2 text-xs text-gray-500">
          <span>Tool calls: {relevantToolCalls.length}</span>
          <span>Engines: {toolGroups.length}</span>
          {selectedNode && <span>Findings: {selectedNode.findings}</span>}
        </div>
      </div>

      {/* Tool call timeline */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {relevantToolCalls.slice(0, 200).map((tc, i) => (
          <motion.div
            key={tc.id}
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: i * 0.01 }}
            className="flex items-start gap-2 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <div
              className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
              style={{ background: getToolColor(tc.tool) }}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                  style={{
                    background: `${getToolColor(tc.tool)}20`,
                    color: getToolColor(tc.tool),
                  }}
                >
                  {tc.tool}
                </span>
                <span className="text-[10px] text-gray-600 ml-auto shrink-0">
                  {tc.duration}s
                </span>
              </div>
              <p className="text-[11px] text-gray-300 mt-0.5 truncate">
                {tc.query}
              </p>
            </div>
          </motion.div>
        ))}
        {relevantToolCalls.length > 200 && (
          <div className="text-xs text-gray-600 text-center py-2">
            + {relevantToolCalls.length - 200} more tool calls
          </div>
        )}
      </div>

      {/* Tool distribution footer */}
      <div className="p-3 border-t border-gray-700 bg-gray-900/30">
        <div className="flex flex-wrap gap-1">
          {toolGroups.slice(0, 8).map(([engine, calls]) => (
            <span
              key={engine}
              className="text-[10px] px-1.5 py-0.5 rounded"
              style={{
                background: `${getToolColor(engine)}20`,
                color: getToolColor(engine),
              }}
            >
              {engine}: {calls.length}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
