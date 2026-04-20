# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native plugins for the Strands Venice agent.

Each plugin uses the Strands SDK Plugin base class with @hook decorators
to replace custom callback code with properly typed lifecycle hooks.
"""

from plugins.budget import BudgetPlugin
from plugins.stream_capture import StreamCapturePlugin
from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_display import ToolDisplayPlugin

__all__ = [
    "BudgetPlugin",
    "StreamCapturePlugin",
    "ThoughtRefinerPlugin",
    "ToolDisplayPlugin",
]
