# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Automated evaluation suite for the Strands Venice agent.

Adapted from the deepagents evals framework pattern with TrajectoryScorer,
two-tier assertions (.success() hard-fail, .expect() soft), and per-category
reporting.  Uses the SDK-native plugins for trajectory capture instead of
LangChain message types.
"""
