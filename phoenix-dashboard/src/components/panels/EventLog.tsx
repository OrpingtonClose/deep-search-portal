// ============================================================
// Event Log Panel – Scrollable SSE event stream
// ============================================================

import { useRef, useEffect, useMemo } from 'react';
import { useStore } from '../../store/useStore';

const EVENT_COLORS: Record<string, string> = {
  'Phase': '#3b82f6',
  'Tree': '#8b5cf6',
  'Querying': '#06b6d4',
  'Reading': '#a855f7',
  'Spawning': '#f59e0b',
  'Investigating': '#10b981',
  'findings': '#ec4899',
  'Admission': '#14b8a6',
  'Cross-check': '#6366f1',
  'Synthesis': '#f97316',
  'Drafting': '#6b7280',
  'Persisting': '#84cc16',
  'Stored': '#84cc16',
  'Decomposition': '#eab308',
  'Critic': '#d946ef',
  'Final': '#d946ef',
  'Re-research': '#ef4444',
  'keepalive': '#374151',
};

function getEventColor(content: string): string {
  for (const [key, color] of Object.entries(EVENT_COLORS)) {
    if (content.includes(key)) return color;
  }
  return '#9ca3af';
}

export function EventLog() {
  const events = useStore(s => s.events);
  const searchQuery = useStore(s => s.searchQuery);
  const setSearchQuery = useStore(s => s.setSearchQuery);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filteredEvents = useMemo(() => {
    if (!searchQuery) return events;
    const q = searchQuery.toLowerCase();
    return events.filter(e => e.content.toLowerCase().includes(q));
  }, [events, searchQuery]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredEvents.length]);

  return (
    <div className="h-full flex flex-col">
      {/* Search bar */}
      <div className="p-2 border-b border-gray-700">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Filter events..."
          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <div className="text-[10px] text-gray-600 mt-1">
          {filteredEvents.length} / {events.length} events
        </div>
      </div>

      {/* Event list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {filteredEvents.map((evt) => {
          const color = getEventColor(evt.content);
          const isDrafting = evt.content.includes('Drafting synthesis');
          return (
            <div
              key={evt.id}
              className={`px-2 py-1 rounded text-[11px] font-mono leading-tight hover:bg-gray-800/50 transition-colors ${isDrafting ? 'opacity-30' : ''}`}
              style={{ borderLeft: `2px solid ${color}` }}
            >
              <span className="text-gray-600 mr-2">#{evt.lineNumber}</span>
              <span style={{ color }}>{evt.content}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
