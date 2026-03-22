// ============================================================
// Dashboard Layout – Main content area with graph + side panels
// ============================================================

import { useStore } from '../../store/useStore';
import { PipelineDAG } from '../graph/PipelineDAG';
import { ResearchTree } from '../graph/ResearchTree';
import { SubagentView } from '../graph/SubagentView';
import { MetricsPanel } from '../panels/MetricsPanel';
import { EventLog } from '../panels/EventLog';
import { ReactFlowProvider } from '@xyflow/react';
import { useState } from 'react';
import { BarChart3, ScrollText, ChevronLeft, ChevronRight as ChevronRightIcon } from 'lucide-react';

type SidePanel = 'metrics' | 'events';

export function Dashboard() {
  const drilldown = useStore(s => s.drilldown);
  const sseFileName = useStore(s => s.sseFileName);
  const [sidePanel, setSidePanel] = useState<SidePanel>('metrics');
  const [sidePanelOpen, setSidePanelOpen] = useState(true);

  if (!sseFileName) {
    return <EmptyState />;
  }

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Main graph area */}
      <div className="flex-1 relative">
        <ReactFlowProvider>
          {drilldown.level === 'pipeline' && <PipelineDAG />}
          {drilldown.level === 'tree' && <ResearchTree />}
          {drilldown.level === 'subagent' && <SubagentView />}
        </ReactFlowProvider>
      </div>

      {/* Side panel toggle */}
      <button
        onClick={() => setSidePanelOpen(!sidePanelOpen)}
        className="w-6 bg-gray-800 border-x border-gray-700 flex items-center justify-center hover:bg-gray-700 transition-colors shrink-0"
      >
        {sidePanelOpen ? (
          <ChevronRightIcon className="w-3.5 h-3.5 text-gray-400" />
        ) : (
          <ChevronLeft className="w-3.5 h-3.5 text-gray-400" />
        )}
      </button>

      {/* Side panel */}
      {sidePanelOpen && (
        <div className="w-80 lg:w-96 bg-gray-900/50 border-l border-gray-700 flex flex-col shrink-0">
          {/* Panel tabs */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setSidePanel('metrics')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-xs transition-colors ${
                sidePanel === 'metrics'
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-400/5'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Metrics
            </button>
            <button
              onClick={() => setSidePanel('events')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-xs transition-colors ${
                sidePanel === 'events'
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-400/5'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <ScrollText className="w-3.5 h-3.5" />
              Events ({useStore.getState().events.length})
            </button>
          </div>

          {/* Panel content */}
          <div className="flex-1 overflow-hidden">
            {sidePanel === 'metrics' && <MetricsPanel />}
            {sidePanel === 'events' && <EventLog />}
          </div>
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-600/20 border border-blue-500/30 flex items-center justify-center">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
            P
          </div>
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Phoenix Dashboard</h2>
        <p className="text-sm text-gray-400 mb-6 leading-relaxed">
          Visualize persistent deep research pipeline execution as an interactive,
          drillable graph. Upload an SSE stream file and optionally a proxy log to begin.
        </p>
        <div className="grid grid-cols-2 gap-3 text-left">
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
            <div className="text-xs font-semibold text-blue-400 mb-1">SSE Stream</div>
            <div className="text-[10px] text-gray-500">
              The OpenAI-compatible SSE response from the persistent proxy (test1_response.txt)
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
            <div className="text-xs font-semibold text-purple-400 mb-1">Proxy Log</div>
            <div className="text-[10px] text-gray-500">
              Server-side structured log with subagent details (persistent-proxy.log)
            </div>
          </div>
        </div>
        <div className="mt-6 text-[10px] text-gray-600">
          Pipeline DAG → Research Tree → Subagent Trace → Conditions
        </div>
      </div>
    </div>
  );
}
