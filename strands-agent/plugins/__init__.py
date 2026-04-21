# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native plugins for the Strands Venice agent.

Each plugin uses the Strands SDK Plugin base class with @hook decorators
to replace custom callback code with properly typed lifecycle hooks.

- BudgetPlugin: tool-call budget tracking and enforcement
- StreamCapturePlugin: captures streaming events for SSE forwarding
- ThoughtRefinerPlugin: refines agent thinking into prose
- ToolDisplayPlugin: descriptive tool labels for the UI
- KnowledgePlugin: cross-conversation knowledge persistence
  (BeforeInvocation / AfterToolCall / AfterInvocation hooks + @tool methods)
- ToolRouterPlugin: query-aware tool routing (BeforeInvocationEvent)
- ToolAuditPlugin: post-invocation tool usage verification (AfterInvocationEvent)
"""

from plugins.budget import BudgetPlugin
from plugins.knowledge import KnowledgePlugin
from plugins.stream_capture import StreamCapturePlugin
from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_audit import ToolAuditPlugin
from plugins.tool_display import ToolDisplayPlugin
from plugins.tool_router import ToolRouterPlugin

__all__ = [
    "BudgetPlugin",
    "KnowledgePlugin",
    "StreamCapturePlugin",
    "ThoughtRefinerPlugin",
    "ToolAuditPlugin",
    "ToolDisplayPlugin",
    "ToolRouterPlugin",
]
