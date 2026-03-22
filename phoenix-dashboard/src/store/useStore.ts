// ============================================================
// Zustand Store – Single source of truth for dashboard state
// ============================================================

import { create } from 'zustand';
import type {
  PipelineNode,
  PipelineEdge,
  ResearchTreeNode,
  ToolCall,
  SubagentInfo,
  RunMetrics,
  SSEEvent,
  LogEvent,
  DrilldownState,
} from '../types';
import { parseSSEStream } from '../parsers/sseParser';
import { parseLogFile } from '../parsers/logParser';

interface DashboardStore {
  // --- Data ---
  pipelineNodes: PipelineNode[];
  pipelineEdges: PipelineEdge[];
  treeNodes: ResearchTreeNode[];
  toolCalls: ToolCall[];
  subagents: SubagentInfo[];
  metrics: RunMetrics | null;
  events: SSEEvent[];
  logEvents: LogEvent[];
  totalLines: number;

  // --- UI State ---
  drilldown: DrilldownState;
  isLoading: boolean;
  sseFileName: string | null;
  logFileName: string | null;
  playbackProgress: number; // 0-1
  isPlaying: boolean;
  playbackSpeed: number; // events per frame
  searchQuery: string;
  selectedEventIndex: number | null;

  // --- Actions ---
  loadSSE: (text: string, fileName: string) => void;
  loadLog: (text: string, fileName: string) => void;
  setDrilldown: (state: Partial<DrilldownState>) => void;
  navigateBack: () => void;
  setPlaybackProgress: (p: number) => void;
  setIsPlaying: (v: boolean) => void;
  setPlaybackSpeed: (s: number) => void;
  setSearchQuery: (q: string) => void;
  setSelectedEventIndex: (i: number | null) => void;
  reset: () => void;
}

const initialDrilldown: DrilldownState = {
  level: 'pipeline',
  iteration: 0,
};

export const useStore = create<DashboardStore>((set, get) => ({
  // --- Initial Data ---
  pipelineNodes: [],
  pipelineEdges: [],
  treeNodes: [],
  toolCalls: [],
  subagents: [],
  metrics: null,
  events: [],
  logEvents: [],
  totalLines: 0,

  // --- Initial UI ---
  drilldown: { ...initialDrilldown },
  isLoading: false,
  sseFileName: null,
  logFileName: null,
  playbackProgress: 1,
  isPlaying: false,
  playbackSpeed: 10,
  searchQuery: '',
  selectedEventIndex: null,

  // --- Actions ---
  loadSSE: (text, fileName) => {
    set({ isLoading: true });
    // Parse in next tick to not block UI
    setTimeout(() => {
      const parsed = parseSSEStream(text);
      set({
        pipelineNodes: parsed.pipelineNodes,
        pipelineEdges: parsed.pipelineEdges,
        treeNodes: parsed.treeNodes,
        toolCalls: parsed.toolCalls,
        metrics: parsed.metrics,
        events: parsed.events,
        totalLines: parsed.totalLines,
        sseFileName: fileName,
        isLoading: false,
        playbackProgress: 1,
        drilldown: { ...initialDrilldown },
      });
    }, 0);
  },

  loadLog: (text, fileName) => {
    setTimeout(() => {
      const parsed = parseLogFile(text);
      set({
        subagents: parsed.subagents,
        logEvents: parsed.logEvents,
        logFileName: fileName,
      });
    }, 0);
  },

  setDrilldown: (partial) => {
    const current = get().drilldown;
    set({ drilldown: { ...current, ...partial } });
  },

  navigateBack: () => {
    const current = get().drilldown;
    switch (current.level) {
      case 'condition':
        set({ drilldown: { ...current, level: 'subagent', selectedSubagent: undefined } });
        break;
      case 'subagent':
        set({ drilldown: { ...current, level: 'tree', selectedTreeNode: undefined } });
        break;
      case 'tree':
        set({ drilldown: { ...current, level: 'pipeline', selectedPhase: undefined } });
        break;
      default:
        break;
    }
  },

  setPlaybackProgress: (p) => set({ playbackProgress: p }),
  setIsPlaying: (v) => set({ isPlaying: v }),
  setPlaybackSpeed: (s) => set({ playbackSpeed: s }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setSelectedEventIndex: (i) => set({ selectedEventIndex: i }),

  reset: () => set({
    pipelineNodes: [],
    pipelineEdges: [],
    treeNodes: [],
    toolCalls: [],
    subagents: [],
    metrics: null,
    events: [],
    logEvents: [],
    totalLines: 0,
    drilldown: { ...initialDrilldown },
    sseFileName: null,
    logFileName: null,
    playbackProgress: 1,
    isPlaying: false,
    searchQuery: '',
    selectedEventIndex: null,
  }),
}));
