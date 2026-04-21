# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the query domain classifier and tool-to-domain mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the strands-agent source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.domains import (
    ACADEMIC,
    ALL_DOMAINS,
    DOMAIN_GUIDANCE,
    DOMAIN_SKILLS,
    DOMAIN_TOOLS,
    FINANCIAL,
    FORUM,
    GENERAL,
    GOVERNMENT,
    OSINT,
    PRACTITIONER,
    PREPRINT,
    YOUTUBE,
    DomainMatch,
    classify_query,
)


class TestClassifyQuery:
    """Test classify_query returns correct domains for various queries."""

    def test_academic_query(self) -> None:
        match = classify_query("find papers on GLP-1 receptor pharmacokinetics")
        assert ACADEMIC in match.domains
        assert match.primary == ACADEMIC

    def test_practitioner_query(self) -> None:
        match = classify_query("best tren dosage protocol for recomp cycle")
        assert PRACTITIONER in match.domains
        assert match.primary == PRACTITIONER

    def test_government_query(self) -> None:
        match = classify_query("clinical trial results for semaglutide FDA approval")
        assert GOVERNMENT in match.domains

    def test_financial_query(self) -> None:
        match = classify_query("SEC 10-K filing analysis for startup revenue growth")
        assert FINANCIAL in match.domains

    def test_youtube_query(self) -> None:
        match = classify_query("find YouTube channels about bodybuilding protocols")
        assert YOUTUBE in match.domains

    def test_osint_query(self) -> None:
        match = classify_query("find censored content archived on wayback machine")
        assert OSINT in match.domains

    def test_forum_query(self) -> None:
        match = classify_query("MesoRx forum thread about vendor reviews")
        assert FORUM in match.domains

    def test_preprint_query(self) -> None:
        match = classify_query("latest bioRxiv preprints on CRISPR gene editing")
        assert PREPRINT in match.domains

    def test_general_query(self) -> None:
        match = classify_query("what time is it in Tokyo")
        assert match.primary == GENERAL
        assert match.domains == (GENERAL,)

    def test_multi_domain_query(self) -> None:
        """Query matching multiple domains returns all matches."""
        match = classify_query(
            "find PubMed papers and forum experience reports on tren bloodwork"
        )
        assert len(match.domains) >= 2
        # Should match both academic (PubMed, papers) and practitioner (tren, bloodwork)
        assert ACADEMIC in match.domains or PRACTITIONER in match.domains

    def test_empty_query(self) -> None:
        match = classify_query("")
        assert match.primary == GENERAL

    def test_domain_match_is_frozen(self) -> None:
        match = classify_query("find papers")
        with pytest.raises(AttributeError):
            match.primary = "something"  # type: ignore[misc]


class TestDomainMappings:
    """Test that domain mappings are complete and consistent."""

    def test_all_domains_have_tools(self) -> None:
        for domain in ALL_DOMAINS:
            assert domain in DOMAIN_TOOLS, f"missing tools for domain {domain}"
            assert len(DOMAIN_TOOLS[domain]) > 0, f"empty tools for domain {domain}"

    def test_all_domains_have_guidance(self) -> None:
        for domain in ALL_DOMAINS:
            assert domain in DOMAIN_GUIDANCE, f"missing guidance for domain {domain}"
            assert len(DOMAIN_GUIDANCE[domain]) > 0, f"empty guidance for domain {domain}"

    def test_all_domains_have_skill_mapping(self) -> None:
        for domain in ALL_DOMAINS:
            assert domain in DOMAIN_SKILLS, f"missing skill mapping for domain {domain}"

    def test_academic_tools_include_core(self) -> None:
        tools = DOMAIN_TOOLS[ACADEMIC]
        assert "openalex_search" in tools
        assert "search_pubmed" in tools
        assert "check_retraction" in tools

    def test_practitioner_tools_include_forums(self) -> None:
        tools = DOMAIN_TOOLS[PRACTITIONER]
        assert "forum_search" in tools
        assert "forum_deep_dive" in tools

    def test_government_tools_include_clinical_trials(self) -> None:
        tools = DOMAIN_TOOLS[GOVERNMENT]
        assert "search_clinical_trials" in tools
        assert "search_fda_adverse_events" in tools

    def test_youtube_tools_include_transcripts(self) -> None:
        tools = DOMAIN_TOOLS[YOUTUBE]
        assert "youtube_download_transcript" in tools
        assert "search_youtube" in tools

    def test_skill_names_match_skill_directories(self) -> None:
        """Verify skill names map to actual skill directories."""
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        for domain, skill_name in DOMAIN_SKILLS.items():
            if skill_name is not None:
                skill_path = skills_dir / skill_name / "SKILL.md"
                assert skill_path.exists(), (
                    f"domain {domain} maps to skill '{skill_name}' "
                    f"but {skill_path} does not exist"
                )
