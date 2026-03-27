"""Tests for claim and fact extraction parsing logic."""
import pytest
from src.modules.claim_extractor import _parse_claim_list
from src.modules.fact_extractor import _parse_fact_list


class TestParseClaimList:
    def test_clean_json_array(self):
        raw = '["Claim one.", "Claim two.", "Claim three."]'
        result = _parse_claim_list(raw)
        assert result == ["Claim one.", "Claim two.", "Claim three."]

    def test_json_with_fences(self):
        raw = '```json\n["Claim A", "Claim B"]\n```'
        result = _parse_claim_list(raw)
        assert "Claim A" in result
        assert "Claim B" in result

    def test_empty_array(self):
        result = _parse_claim_list("[]")
        assert result == []

    def test_malformed_falls_back_gracefully(self):
        raw = "- This is claim one\n- This is claim two and it is long enough"
        result = _parse_claim_list(raw)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_strips_whitespace_from_claims(self):
        raw = '["  spaced claim  ", "another  "]'
        result = _parse_claim_list(raw)
        assert all(c == c.strip() for c in result)


class TestParseFactList:
    def test_clean_json_array(self):
        raw = '["Fact one.", "Fact two."]'
        result = _parse_fact_list(raw)
        assert len(result) == 2

    def test_filters_very_short_entries(self):
        raw = '["ok", "This is a real fact with enough content to count."]'
        result = _parse_fact_list(raw)
        # Short entries (< 10 chars) should be filtered out
        assert all(len(f) >= 10 for f in result)
