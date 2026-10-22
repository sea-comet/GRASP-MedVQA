"""Output format shared by the prompt, the rewards and the SFT target.

A response is ``<REASONING> ... </REASONING><answer> ... </answer>``. Every
module reads the tags and the length limits from here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

MIN_REASONING_WORDS = 8
MAX_REASONING_WORDS = 40
REPETITION_NGRAM_SIZE = 4

_REASONING_PATTERN = re.compile(
    rf"{re.escape(REASONING_START)}(.*?){re.escape(REASONING_END)}", flags=re.DOTALL
)
_SOLUTION_PATTERN = re.compile(
    rf"{re.escape(SOLUTION_START)}(.*?){re.escape(SOLUTION_END)}", flags=re.DOTALL
)
_STRICT_PATTERN = re.compile(
    rf"^\s*{re.escape(REASONING_START)}(.*?){re.escape(REASONING_END)}"
    rf"\s*{re.escape(SOLUTION_START)}(.*?){re.escape(SOLUTION_END)}\s*$",
    flags=re.DOTALL,
)
_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)$")
_CLAUSE_END = re.compile(r"[,;:](?:[\"')\]]*)$")


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def _normalized_word(word: str) -> str:
    return re.sub(r"(^\W+|\W+$)", "", word).casefold()


def first_repeated_ngram_start(text: str, *, ngram_size: int = REPETITION_NGRAM_SIZE) -> int | None:
    """Word index where the first repeated n-gram begins, or None."""

    words = [_normalized_word(word) for word in normalize_whitespace(text).split()]
    ngram_size = max(int(ngram_size), 2)
    if len(words) < 2 * ngram_size:
        return None
    seen: set[tuple[str, ...]] = set()
    for index in range(len(words) - ngram_size + 1):
        ngram = tuple(words[index : index + ngram_size])
        if not all(ngram):
            continue
        if ngram in seen:
            return index
        seen.add(ngram)
    return None


def has_obvious_repetition(text: str, *, ngram_size: int = REPETITION_NGRAM_SIZE) -> bool:
    return first_repeated_ngram_start(text, ngram_size=ngram_size) is not None


def _raw_reasoning_candidate(completion: str) -> str:
    """Recover possible reasoning even when tags are missing or incomplete."""

    text = str(completion or "")
    start = text.find(REASONING_START)
    candidate = text[start + len(REASONING_START) :] if start >= 0 else text
    stops = [
        position
        for marker in (REASONING_END, SOLUTION_START, SOLUTION_END)
        if (position := candidate.find(marker)) >= 0
    ]
    if stops:
        candidate = candidate[: min(stops)]
    for marker in (REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END):
        candidate = candidate.replace(marker, " ")
    return normalize_whitespace(candidate)


def salvage_reasoning(
    text: str,
    *,
    max_words: int = MAX_REASONING_WORDS,
    min_words: int = MIN_REASONING_WORDS,
    ngram_size: int = REPETITION_NGRAM_SIZE,
) -> str:
    """Longest non-repeating prefix within ``max_words``, cut at the last
    sentence (or clause) boundary that is at least ``min_words`` long."""

    words = normalize_whitespace(text).split()
    if not words:
        return ""
    cap = min(len(words), max(int(max_words), 1))
    repeated_at = first_repeated_ngram_start(" ".join(words), ngram_size=ngram_size)
    if repeated_at is not None:
        cap = min(cap, max(repeated_at, 1))
    floor = min(cap, max(int(min_words), 1))
    boundary = cap
    for pattern in (_SENTENCE_END, _CLAUSE_END):
        aligned = [
            index
            for index, word in enumerate(words[:cap], start=1)
            if index >= floor and pattern.search(word)
        ]
        if aligned:
            boundary = aligned[-1]
            break
    return " ".join(words[: max(boundary, 1)])


@dataclass(frozen=True)
class ReasoningContext:
    text: str
    source: str  # "valid", "salvaged" or "empty"

    @property
    def usable(self) -> bool:
        return self.source in {"valid", "salvaged"} and bool(self.text.strip())


def reasoning_context_from_completion(
    completion: str,
    *,
    max_words: int = MAX_REASONING_WORDS,
    min_words: int = MIN_REASONING_WORDS,
    ngram_size: int = REPETITION_NGRAM_SIZE,
) -> ReasoningContext:
    """The rollout's own reasoning if it is valid, else a repaired prefix,
    else an empty (unusable) context."""

    text = str(completion or "")
    min_words = max(int(min_words), 1)
    matches = _REASONING_PATTERN.findall(text)
    original = normalize_whitespace(matches[0]) if len(matches) == 1 else ""
    if (
        len(matches) == 1
        and min_words <= len(original.split()) <= int(max_words)
        and not has_obvious_repetition(original, ngram_size=ngram_size)
    ):
        return ReasoningContext(text=original, source="valid")

    candidate = original if original else _raw_reasoning_candidate(text)
    repaired = salvage_reasoning(
        candidate, max_words=max_words, min_words=min_words, ngram_size=ngram_size
    )
    if len(repaired.split()) >= min_words:
        return ReasoningContext(text=repaired, source="salvaged")
    return ReasoningContext(text="", source="empty")


def analyze_completion_format(
    completion: str,
    *,
    truncated: bool = False,
    is_close_ended: bool = False,
    min_reasoning_words: int = MIN_REASONING_WORDS,
    max_reasoning_words: int = MAX_REASONING_WORDS,
    ngram_size: int = REPETITION_NGRAM_SIZE,
) -> dict[str, object]:
    """Parse one response into the items the format reward scores."""

    text = str(completion or "")
    reasoning_matches = _REASONING_PATTERN.findall(text)
    solution_matches = _SOLUTION_PATTERN.findall(text)
    reasoning = normalize_whitespace(reasoning_matches[0]) if len(reasoning_matches) == 1 else ""
    solution = normalize_whitespace(solution_matches[0]) if len(solution_matches) == 1 else ""
    reasoning_words = len(reasoning.split())
    solution_words = len(solution.split())
    repetition_text = reasoning or _raw_reasoning_candidate(text)
    repeated = (
        has_obvious_repetition(repetition_text, ngram_size=ngram_size) if repetition_text else False
    )
    reasoning_length_valid = int(min_reasoning_words) <= reasoning_words <= int(max_reasoning_words)
    # One word ("yes"/"no") for close-ended questions, one to three otherwise.
    solution_length_valid = solution_words == 1 if is_close_ended else 1 <= solution_words <= 3
    tag_counts_valid = all(
        text.count(marker) == 1
        for marker in (REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END)
    )
    strict_full_format_valid = (
        _STRICT_PATTERN.fullmatch(text) is not None
        and tag_counts_valid
        and reasoning_length_valid
        and solution_length_valid
        and not repeated
        and not bool(truncated)
    )
    return {
        "reasoning": reasoning,
        "solution": solution,
        "reasoning_tag_valid": len(reasoning_matches) == 1,
        "solution_tag_valid": len(solution_matches) == 1,
        "reasoning_nonempty": bool(reasoning),
        "reasoning_length_valid": reasoning_length_valid,
        "solution_length_valid": solution_length_valid,
        "repetition_detected": repeated,
        "strict_full_format_valid": strict_full_format_valid,
        "truncated": bool(truncated),
    }
