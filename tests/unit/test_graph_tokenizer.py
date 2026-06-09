"""Unit tests for app.retrieve.tokenizer.question_tokens (Fix A).

These drive the token-boundary seed matching that replaces the substring
CONTAINS match. The canonical regression: a short Korean entity like "원장"
must NOT be produced as a token of a question that only contains "부원장".
"""
from __future__ import annotations

from app.retrieve.tokenizer import question_tokens


# ---------- empty / whitespace --------------------------------------------

def test_empty_string_returns_empty_list():
    assert question_tokens("") == []


def test_whitespace_only_returns_empty_list():
    assert question_tokens("   \t\n　") == []


# ---------- basic inclusion -----------------------------------------------

def test_plain_korean_word_included():
    assert "출장" in question_tokens("출장")


def test_raw_token_preserved_alongside_stripped():
    tokens = question_tokens("원장의 성명은?")
    assert "원장의" in tokens  # raw form kept
    assert "원장" in tokens     # josa-stripped form also kept


# ---------- josa stripping ------------------------------------------------

def test_josa_의_stripped():
    assert "원장" in question_tokens("원장의 성명은?")


def test_josa_은_stripped():
    assert "출장" in question_tokens("출장은 언제인가?")


def test_josa_이_stripped():
    assert "이지환" in question_tokens("이지환이 결재한다")


def test_josa_을_stripped():
    assert "승인" in question_tokens("승인을 받는다")


def test_longest_josa_으로_stripped_whole_not_partial():
    # '으로' must be stripped as a unit, not leave a trailing '으'.
    tokens = question_tokens("자동으로 처리된다")
    assert "자동" in tokens
    assert "자동으" not in tokens


# ---------- compound (stacked) josa — must reduce to the base entity --------
# Phase 1's CONTAINS match found the entity inside any trailing particles;
# token-set mode must recover the base across two-particle stacks too.

def test_compound_josa_에서부터_recovers_base():
    assert "회사" in question_tokens("회사에서부터 시작되었다")


def test_compound_josa_에서의_recovers_base():
    assert "출장" in question_tokens("출장에서의 비용 한도는?")


def test_compound_josa_로부터_recovers_base():
    assert "학교" in question_tokens("학교로부터 배운다")


def test_compound_josa_에서는_recovers_base():
    assert "회사" in question_tokens("회사에서는 가능한가?")


# ---------- normalization -------------------------------------------------

def test_nfkc_normalizes_fullwidth_chars():
    # Fullwidth 'ＡＢ' → NFKC 'AB' → casefold 'ab' (len 2, survives min_len).
    assert "ab" in question_tokens("ＡＢ")


def test_casefold_lowercases_ascii():
    tokens = question_tokens("KPRI 정책")
    assert "kpri" in tokens
    assert "KPRI" not in tokens


def test_min_len_excludes_short_tokens():
    tokens = question_tokens("a 이 가 출장", min_len=2)
    assert all(len(t) >= 2 for t in tokens)
    assert "출장" in tokens


# ---------- the canonical false-positive regressions ----------------------

def test_원장_not_produced_from_부원장_question():
    # The Phase 1 bug: substring match surfaced '원장' inside '부원장'.
    tokens = question_tokens("부원장의 출장 승인 권한은?")
    assert "원장" not in tokens
    assert "부원장" in tokens


def test_출장_not_produced_from_출장비_compound():
    tokens = question_tokens("출장비 결재는 누가 대행하는가?")
    assert "출장" not in tokens
    assert "출장비" in tokens


def test_출장_produced_when_josa_attached():
    assert "출장" in question_tokens("출장의 일일 식비 한도는 얼마인가?")


# ---------- shape / purity ------------------------------------------------

def test_result_is_sorted_list():
    result = question_tokens("원장 출장 승인")
    assert isinstance(result, list)
    assert result == sorted(result)


def test_result_is_deduplicated():
    # '출장' appears twice (raw + via '출장의' strip) but only once in output.
    result = question_tokens("출장 출장의")
    assert result.count("출장") == 1


def test_new_call_returns_new_object():
    a = question_tokens("원장 출장")
    b = question_tokens("원장 출장")
    assert a == b
    assert a is not b
