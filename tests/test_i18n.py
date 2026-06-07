"""Unit tests for src.frontend.i18n (M8.5d).

Three test groups:

* :func:`validate_consistency` — all 3 languages have the same keys
* :func:`t` — fallback to English, then to the key
* :func:`t` — format placeholders are filled in
"""

from __future__ import annotations

import pytest

from src.frontend.i18n import (
    DEFAULT_LANG,
    LANG_LABELS,
    SUPPORTED_LANGS,
    available_keys,
    current_language,
    t,
    validate_consistency,
)


class TestLanguageMetadata:
    def test_default_lang_is_english(self):
        assert DEFAULT_LANG == "en"

    def test_supported_langs(self):
        assert "en" in SUPPORTED_LANGS
        assert "zh" in SUPPORTED_LANGS
        assert "zh-HK" in SUPPORTED_LANGS

    def test_lang_labels_cover_all(self):
        for code in SUPPORTED_LANGS:
            assert code in LANG_LABELS
            assert len(LANG_LABELS[code]) > 0


class TestConsistency:
    def test_all_languages_have_same_keys(self):
        missing = validate_consistency()
        assert missing == {}, f"Missing translations: {missing}"

    def test_english_has_at_least_50_keys(self):
        # We added ~50+ keys; should not regress.
        assert len(available_keys("en")) >= 50

    def test_each_language_has_same_count(self):
        counts = {lang: len(available_keys(lang)) for lang in SUPPORTED_LANGS}
        # All counts should match
        unique_counts = set(counts.values())
        assert len(unique_counts) == 1, f"Key counts differ: {counts}"


class TestLookup:
    def test_known_key_english(self):
        assert t("app.title", "en").startswith("CausalCredit")

    def test_known_key_chinese_simplified(self):
        # Should be a Chinese string (any CJK char)
        out = t("app.title", "zh")
        assert any("一" <= c <= "鿿" for c in out), f"expected Chinese, got {out!r}"

    def test_known_key_chinese_hk(self):
        out = t("app.title", "zh-HK")
        # zh-HK uses 繁體 — different characters from 简体 for at least
        # one of {推斷 vs 推断, 評分 vs 评分}. The simplest assertion is
        # "contains CJK and is non-empty".
        assert any("一" <= c <= "鿿" for c in out)

    def test_unknown_key_returns_key_itself(self):
        # The frontend must never crash on an untranslated key.
        assert t("not.a.real.key", "en") == "not.a.real.key"
        assert t("not.a.real.key", "zh") == "not.a.real.key"
        assert t("not.a.real.key", "zh-HK") == "not.a.real.key"

    def test_unknown_lang_falls_back_to_english(self):
        out = t("app.title", "xx-YY")
        assert out == t("app.title", "en")

    def test_default_lang_fallback(self):
        # t(key) with no lang → English
        assert t("app.title") == t("app.title", "en")


class TestFormatPlaceholders:
    def test_single_placeholder_filled(self):
        out = t("app.sidebar_caption", "en", n_features=42)
        assert "42 features" in out

    def test_multiple_placeholders_filled(self):
        out = t(
            "causal.treatments_outcome", "en",
            treatments="AMT_CREDIT",
            outcome="default",
            n_nodes=10, n_edges=15,
        )
        assert "AMT_CREDIT" in out
        assert "default" in out
        assert "10" in out
        assert "15" in out

    def test_missing_kwarg_leaves_placeholder(self):
        out = t("app.sidebar_caption", "en")
        # {n_features} is unfilled — the function should leave the literal
        # placeholder rather than crash.
        assert "{n_features}" in out

    def test_format_works_in_all_languages(self):
        for lang in SUPPORTED_LANGS:
            out = t("causal.treatments_outcome", lang,
                    treatments="T", outcome="Y", n_nodes=1, n_edges=1)
            # No curly braces should remain (placeholders all filled)
            assert "{" not in out, f"Unfilled placeholder in {lang}: {out!r}"


class TestCurrentLanguage:
    def test_returns_default_when_no_session(self, monkeypatch):
        # Without Streamlit session, should return default
        lang = current_language()
        assert lang in SUPPORTED_LANGS
