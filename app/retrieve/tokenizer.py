"""Korean-aware question tokenizer for graph seed matching (Fix A).

Phase 1 eval showed graph+vector underperforming vector-only because
``graph_search`` matched seed entities with ``$question CONTAINS toLower(name)``
— a substring test that surfaced short Korean entities (e.g. "원장") whenever
they appeared *inside* an unrelated word ("부원장") or compound ("출장비").

This module splits a question into normalized tokens so seeds can be matched
with exact token-set equality (``toLower(name) IN $tokens``) instead. For each
whitespace/punctuation-delimited token we emit BOTH the raw normalized form and
its josa-stripped form, so an entity stored as "원장" still matches a question
token "원장의" without re-introducing substring false positives.

Limitation: token-set mode only seeds entities whose name is a single
whitespace-delimited token. A multi-word entity name (e.g. "Bob Park") is
never a member of the token set, so it is not seeded in this mode — the
default substring mode in ``graph.py`` is the one that handles such names.

Pure module: no ``app.*`` imports, no I/O, deterministic sorted output.
"""
from __future__ import annotations

import re
import unicodedata

# Split on whitespace + common ASCII/CJK punctuation, fullwidth space, NBSP.
_SPLIT_RE = re.compile(
    r"[\s()\[\]{}<>?.,!;:\"'`/\\~·…“”‘’《》〈〉「」『』，。！？、　 ]+"
)

# Korean postpositional particles (josa). Multi-character variants are listed
# BEFORE their single-character suffixes so the regex strips the longest match
# first (e.g. "으로" before "로", "에서" before "서") — otherwise "자동으로"
# would strip only "로" and leave a stray "자동으". A non-capturing group is
# used since re.sub() replaces the whole match. question_tokens() applies this
# repeatedly to a fixpoint so stacked particles ("회사에서부터" → "회사") reduce
# fully to the base entity.
_JOSA_RE = re.compile(
    r"(?:"
    r"으로부터|로부터|으로서|으로써|에게서|한테서|에서|에게|한테|으로|부터|까지|보다|처럼"
    r"|마다|조차|밖에|이라|라고|이나"
    r"|은|는|이|가|을|를|의|에|도|만|와|과|로|랑|나|며|고|라|야"
    r")$"
)

_MIN_LEN_DEFAULT = 2


def question_tokens(text: str, min_len: int = _MIN_LEN_DEFAULT) -> list[str]:
    """Return sorted, deduplicated, normalized tokens from a question.

    Each token is NFKC-normalized and casefolded. For every delimited token we
    add both its raw form and its josa-stripped form. Tokens shorter than
    ``min_len`` are dropped. Returns a new list on every call.
    """
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: set[str] = set()
    for raw in _SPLIT_RE.split(normalized):
        if not raw:
            continue
        tokens.add(raw)
        # Strip trailing particles repeatedly so stacked josa reduce to the
        # base form ("회사에서부터" → "회사에서" → "회사").
        cur = raw
        while True:
            stripped = _JOSA_RE.sub("", cur)
            if not stripped or stripped == cur:
                break
            tokens.add(stripped)
            cur = stripped
    return sorted(t for t in tokens if len(t) >= min_len)
