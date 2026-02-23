"""Complexity Analyzer - Analyze task complexity for intelligent routing."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class ComplexityIndicators:
    """Indicators extracted from prompt for complexity analysis."""

    word_count: int
    sentence_count: int
    has_code_blocks: bool
    has_math: bool
    has_reasoning_keywords: bool
    has_creative_keywords: bool
    has_analysis_keywords: bool
    has_simple_keywords: bool
    step_count: int
    question_count: int
    avg_sentence_length: float


class ComplexityAnalyzer:
    """Analyze prompt complexity for routing decisions."""

    REASONING_KEYWORDS: Set[str] = {
        "explain",
        "analyze",
        "compare",
        "contrast",
        "evaluate",
        "reason",
        "logic",
        "deduce",
        "infer",
        "conclude",
        "step by step",
        "think through",
        "walk me through",
        "why",
        "how does",
        "what if",
        "consider",
        "advantages",
        "disadvantages",
        "pros and cons",
        "implications",
        "consequences",
        "causes",
        "effects",
        "trade-offs",
        "assess",
        "critique",
        "justify",
    }

    CREATIVE_KEYWORDS: Set[str] = {
        "write",
        "create",
        "generate",
        "compose",
        "draft",
        "story",
        "poem",
        "essay",
        "article",
        "blog",
        "creative",
        "imagine",
        "fiction",
        "narrative",
        "dialogue",
        "script",
        "scenario",
        "brainstorm",
        "design",
        "invent",
        "novel",
        "unique",
        "original",
    }

    ANALYSIS_KEYWORDS: Set[str] = {
        "analyze",
        "examine",
        "investigate",
        "study",
        "data",
        "statistics",
        "trend",
        "pattern",
        "research",
        "review",
        "assess",
        "critique",
        "synthesize",
        "summarize findings",
        "interpret",
        "metrics",
        "performance",
        "benchmark",
        "report",
        "evaluate",
        "measure",
        "quantify",
    }

    SIMPLE_KEYWORDS: Set[str] = {
        "what is",
        "who is",
        "when",
        "where",
        "define",
        "list",
        "name",
        "identify",
        "yes or no",
        "true or false",
        "short",
        "brief",
        "quick",
        "simple",
        "basic",
        "give me",
        "tell me",
        "show me",
    }

    CODE_PATTERNS = [
        r"```[\s\S]*?```",  # Markdown code blocks
        r"`[^`]+`",  # Inline code
        r"^\s{4,}",  # Indented code
    ]

    MATH_PATTERNS = [
        r"\d+\s*[+\-*/=]\s*\d+",  # Basic operations
        r"(?:<=|>=|==|!=|<|>)",  # Comparison operators
        r"\^\d+",  # Exponents
        r"\\[a-zA-Z]+\{",  # LaTeX commands
    ]

    def __init__(self) -> None:
        self._cache: Dict[int, TaskComplexity] = {}

    def analyze(self, prompt: str) -> TaskComplexity:
        cache_key = hash(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        indicators = self._extract_indicators(prompt)
        complexity = self._calculate_complexity(indicators)
        self._cache[cache_key] = complexity
        return complexity

    def _extract_indicators(self, prompt: str) -> ComplexityIndicators:
        prompt_lower = prompt.lower()

        words = prompt.split()
        word_count = len(words)
        sentences = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

        has_code_blocks = any(re.search(pattern, prompt, re.MULTILINE) for pattern in self.CODE_PATTERNS)
        has_math = any(re.search(pattern, prompt) for pattern in self.MATH_PATTERNS)

        has_reasoning_keywords = any(keyword in prompt_lower for keyword in self.REASONING_KEYWORDS)
        has_creative_keywords = any(keyword in prompt_lower for keyword in self.CREATIVE_KEYWORDS)
        has_analysis_keywords = any(keyword in prompt_lower for keyword in self.ANALYSIS_KEYWORDS)
        has_simple_keywords = any(keyword in prompt_lower for keyword in self.SIMPLE_KEYWORDS)

        step_count = len(re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*])\s", prompt))
        question_count = prompt.count("?")

        return ComplexityIndicators(
            word_count=word_count,
            sentence_count=sentence_count,
            has_code_blocks=has_code_blocks,
            has_math=has_math,
            has_reasoning_keywords=has_reasoning_keywords,
            has_creative_keywords=has_creative_keywords,
            has_analysis_keywords=has_analysis_keywords,
            has_simple_keywords=has_simple_keywords,
            step_count=step_count,
            question_count=question_count,
            avg_sentence_length=avg_sentence_length,
        )

    def _calculate_complexity(self, indicators: ComplexityIndicators) -> TaskComplexity:
        score = 0.0

        if indicators.word_count < 10:
            score -= 2
        elif indicators.word_count < 50:
            score += 0
        elif indicators.word_count < 150:
            score += 2
        else:
            score += 4

        if indicators.avg_sentence_length > 20:
            score += 2
        elif indicators.avg_sentence_length > 15:
            score += 1

        if indicators.has_code_blocks:
            score += 3
        if indicators.has_math:
            score += 2

        if indicators.has_reasoning_keywords:
            score += 3
        if indicators.has_analysis_keywords:
            score += 4
        if indicators.has_creative_keywords:
            score += 2
        if indicators.has_simple_keywords:
            score -= 3

        if indicators.step_count > 0:
            score += min(indicators.step_count, 3)
        if indicators.question_count > 1:
            score += min(indicators.question_count, 3)

        if score < 0:
            return TaskComplexity.SIMPLE
        if score < 4:
            return TaskComplexity.MODERATE
        if score < 8:
            return TaskComplexity.COMPLEX
        return TaskComplexity.VERY_COMPLEX

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)