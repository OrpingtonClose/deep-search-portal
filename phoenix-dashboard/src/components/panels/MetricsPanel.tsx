// ============================================================
// Metrics Panel – Charts and stats from the research run
// ============================================================

import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';
import { useStore } from '../../store/useStore';

const CHART_COLORS = [
  '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
  '#06b6d4', '#f97316', '#6366f1', '#14b8a6', '#a855f7',
  '#ef4444', '#84cc16', '#d946ef', '#0ea5e9',
];

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</div>
      <div className="text-lg font-bold text-white mt-0.5">{value}</div>
      {sub && <div className="text-[10px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function MetricsPanel() {
  const metrics = useStore(s => s.metrics);
  const toolCalls = useStore(s => s.toolCalls);
  const treeNodes = useStore(s => s.treeNodes);
  const events = useStore(s => s.events);

  // Tool distribution pie chart data
  const toolPieData = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics.toolCallsByEngine)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([name, value]) => ({ name: name.replace('_search', ''), value }));
  }, [metrics]);

  // Iteration comparison bar chart
  const iterationData = useMemo(() => {
    if (!metrics) return [];
    const data = [];
    for (let i = 0; i <= metrics.totalIterations; i++) {
      const tree = metrics.treeExploration.find(t => t.iteration === i);
      const admission = metrics.admission.find(a => a.iteration === i);
      const cross = metrics.crossCheck.find(c => c.iteration === i);
      data.push({
        name: i === 0 ? 'Pass 1' : `Re-research ${i}`,
        conditions: tree?.atomicConditions || 0,
        admitted: admission?.admitted || 0,
        highConf: cross?.highConfidence || 0,
        lowConf: cross?.lowConfidence || 0,
        toolCalls: tree?.totalToolCalls || 0,
      });
    }
    return data;
  }, [metrics]);

  // Comprehension radar
  const comprehensionRadar = useMemo(() => {
    if (!metrics || metrics.comprehension.length === 0) return [];
    const latest = metrics.comprehension[metrics.comprehension.length - 1];
    return [
      { metric: 'Entities', value: latest.entities },
      { metric: 'Domains', value: latest.domains },
      { metric: 'Implicit Qs', value: latest.implicitQuestions },
      { metric: 'Adjacent', value: latest.adjacentTerritories },
    ];
  }, [metrics]);

  if (!metrics) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Load an SSE stream to view metrics
      </div>
    );
  }

  const latestTree = metrics.treeExploration[metrics.treeExploration.length - 1];
  const latestAdmission = metrics.admission[metrics.admission.length - 1];
  const latestCross = metrics.crossCheck[metrics.crossCheck.length - 1];
  const latestReflection = metrics.reflection[metrics.reflection.length - 1];

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <StatCard
          label="Tree Nodes"
          value={latestTree?.nodesExplored || treeNodes.length}
          sub={`depth ${latestTree?.depthReached || 0}`}
        />
        <StatCard
          label="Conditions"
          value={latestAdmission?.admitted || 0}
          sub={`${latestAdmission?.duplicatesRejected || 0} dupes rejected`}
        />
        <StatCard
          label="Tool Calls"
          value={toolCalls.length}
          sub={`${Object.keys(metrics.toolCallsByEngine).length} engines`}
        />
        <StatCard
          label="Confidence"
          value={`${latestCross?.highConfidence || 0}`}
          sub={`${latestCross?.lowConfidence || 0} low, ${latestCross?.speculative || 0} spec`}
        />
        <StatCard
          label="Iterations"
          value={metrics.totalIterations + 1}
          sub="research passes"
        />
        <StatCard
          label="Quality"
          value={latestReflection?.decompositionQuality.toFixed(1) || 'N/A'}
          sub="decomposition score"
        />
        <StatCard
          label="Events"
          value={events.length}
          sub="SSE events parsed"
        />
        {metrics.entities.length > 0 && (
          <StatCard
            label="Knowledge Graph"
            value={metrics.entities.reduce((s, e) => s + e.newEntities, 0)}
            sub={`${metrics.entities.reduce((s, e) => s + e.newEdges, 0)} edges`}
          />
        )}
      </div>

      {/* Tool distribution */}
      <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-700/50">
        <h4 className="text-xs font-semibold text-gray-400 mb-3">Tool Call Distribution</h4>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={toolPieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={70}
                paddingAngle={2}
                dataKey="value"
              >
                {toolPieData.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                itemStyle={{ color: '#e5e7eb' }}
              />
              <Legend
                formatter={(value) => <span className="text-[10px] text-gray-400">{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Iteration comparison */}
      {iterationData.length > 1 && (
        <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-700/50">
          <h4 className="text-xs font-semibold text-gray-400 mb-3">Iteration Comparison</h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={iterationData}>
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  itemStyle={{ color: '#e5e7eb' }}
                />
                <Bar dataKey="conditions" fill="#3b82f6" name="Conditions" />
                <Bar dataKey="admitted" fill="#10b981" name="Admitted" />
                <Bar dataKey="highConf" fill="#8b5cf6" name="High Conf" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Comprehension radar */}
      {comprehensionRadar.length > 0 && (
        <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-700/50">
          <h4 className="text-xs font-semibold text-gray-400 mb-3">Query Comprehension</h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={comprehensionRadar}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <PolarRadiusAxis tick={{ fontSize: 9, fill: '#6b7280' }} />
                <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
