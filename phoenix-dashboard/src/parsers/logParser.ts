// ============================================================
// Proxy Log Parser
// Parses the persistent-proxy.log structured log format
// into SubagentInfo records for the tree/subagent drilldown
// ============================================================

import type { SubagentInfo, LogEvent } from '../types';

const LOG_LINE_PATTERN = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(INFO|WARNING|ERROR)\] (?:\[(req-[a-z0-9-]+(?:-sa\d+(?:-d\d+)*)?)\] )?(.+)$/;

const SUBAGENT_START_PATTERN = /^Starting subagent: (.+?) \(depth=(\d+)\)$/;
const SUBAGENT_COMPLETE_PATTERN = /^Subagent complete: (\d+) conditions, (\d+) turns, (\d+) tool calls, (\d+) children spawned$/;
const SATURATION_PATTERN = /^Saturation detected \(novelty=([\d.]+)\), stopping early$/;
const MID_TURN_PATTERN = /^Mid-turn admission: (\d+) admitted, (\d+) rejected$/;

/** Parse agent ID into components */
function parseAgentId(agentId: string): { reqId: string; subagentIdx: number; depth: number } {
  const parts = agentId.split('-');
  // e.g., "req-a55ca693-sa0-d1" or "req-a55ca693-sa0"
  let subagentIdx = 0;
  let depth = 0;
  for (const part of parts) {
    if (part.startsWith('sa')) {
      subagentIdx = parseInt(part.slice(2), 10) || 0;
    }
    if (part.startsWith('d')) {
      const dVal = parseInt(part.slice(1), 10);
      if (!isNaN(dVal)) depth = dVal;
    }
  }
  const reqId = parts.slice(0, 2).join('-');
  return { reqId, subagentIdx, depth };
}

export function parseLogFile(text: string): ParsedLog {
  const lines = text.split('\n');
  const logEvents: LogEvent[] = [];
  const subagents: Map<string, SubagentInfo> = new Map();
  let lineNumber = 0;

  for (const line of lines) {
    lineNumber++;
    const match = line.match(LOG_LINE_PATTERN);
    if (!match) continue;

    const [, timestamp, level, agentId, message] = match;
    if (!agentId) continue; // Skip non-agent log lines

    const logEvent: LogEvent = {
      timestamp,
      level: level as LogEvent['level'],
      agentId,
      message,
      lineNumber,
    };
    logEvents.push(logEvent);

    // Extract subagent info
    const startMatch = message.match(SUBAGENT_START_PATTERN);
    if (startMatch) {
      const question = startMatch[1];
      const depth = parseInt(startMatch[2], 10);
      parseAgentId(agentId);
      if (!subagents.has(agentId)) {
        subagents.set(agentId, {
          id: agentId,
          question,
          depth,
          parentSubagentId: findParentId(agentId),
          status: 'active',
          conditions: 0,
          turns: 0,
          toolCalls: 0,
          childrenSpawned: 0,
          novelty: 1.0,
          toolCallDetails: [],
        });
      }
    }

    const completeMatch = message.match(SUBAGENT_COMPLETE_PATTERN);
    if (completeMatch) {
      const sa = subagents.get(agentId);
      if (sa) {
        sa.status = 'completed';
        sa.conditions = parseInt(completeMatch[1], 10);
        sa.turns = parseInt(completeMatch[2], 10);
        sa.toolCalls = parseInt(completeMatch[3], 10);
        sa.childrenSpawned = parseInt(completeMatch[4], 10);
      }
    }

    const saturationMatch = message.match(SATURATION_PATTERN);
    if (saturationMatch) {
      const sa = subagents.get(agentId);
      if (sa) {
        sa.status = 'saturated';
        sa.novelty = parseFloat(saturationMatch[1]);
      }
    }

    const midTurnMatch = message.match(MID_TURN_PATTERN);
    if (midTurnMatch) {
      // Track admission stats per subagent
    }
  }

  return {
    logEvents,
    subagents: Array.from(subagents.values()),
  };
}

function findParentId(agentId: string): string | null {
  // "req-a55ca693-sa0-d1" -> parent is "req-a55ca693-sa0"
  const lastDash = agentId.lastIndexOf('-d');
  if (lastDash > 0 && agentId.slice(lastDash).match(/^-d\d+$/)) {
    return agentId.slice(0, lastDash);
  }
  return null;
}

export interface ParsedLog {
  logEvents: LogEvent[];
  subagents: SubagentInfo[];
}
