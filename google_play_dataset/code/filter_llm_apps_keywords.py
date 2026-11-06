#!/usr/bin/env python3
"""Build an LLM-scoped, auditable app candidate pool.

This stage is a retrieval step, not a final classifier.  It searches Google
Play descriptions and recent-change text for explicit LLM identities,
language-model technology, and AI-plus-language capabilities.  Purely visual,
video, or audio generative-AI matches are written to a separate control pool
instead of being mixed into the primary LLM candidate pool.

Every emitted record contains the rule ids and source snippets that caused it
to be selected.  The resulting candidates should still be screened by the
semantic LLM stage and, ultimately, by APK evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


SCRIPT_VERSION = "llm-keyword-retrieval-v5"
RULESET_VERSION = "llm-keywords-2026-09-16-v5-language-coverage-fixes"
ANALYSIS_SCHEMA_VERSION = "llm-keyword-analysis-v3"
SUMMARY_SCHEMA_VERSION = "llm-keyword-summary-v3"
SELECTION_POLICY_VERSION = "language-features-override-traditional-terms-v2"
CONTEXT_WINDOW_CHARS = 600

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
URL_OR_EMAIL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b\S+@\S+\.\S+\b",
    re.IGNORECASE,
)


def bounded_alternatives(*patterns: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """Compile alternatives with conservative ASCII token boundaries."""
    body = "|".join(f"(?:{pattern})" for pattern in patterns)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", flags)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    tier: str
    pattern: re.Pattern[str]


# High-confidence product/model identities.  Ambiguous bare family words such
# as Claude, Gemini, Llama, Mistral, Gemma, Phi, Falcon, Nova, and Yi are handled
# separately with an AI/language-context guard.
NAMED_LLM_RULES: tuple[Rule, ...] = (
    Rule(
        "named.openai_chatgpt_gpt",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"chat[\s._-]*gpt",
            r"gpt[\s._-]?(?:[2-9](?:\.\d+)?(?:o|mini|turbo|nano|pro|codex)?|4o)",
            r"openai(?:\s+(?:api|platform|assistant|responses?\s+api))?",
            r"azure\s+openai",
            r"openai[\s._-](?:o[134]|codex)",
        ),
    ),
    Rule(
        "named.anthropic_claude",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"anthropic",
            r"claude[\s._-]?(?:\d(?:[\s._-]\d+)?|instant|haiku|sonnet|opus|mythos)",
        ),
    ),
    Rule(
        "named.google_gemini",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"gemini[\s._-]?(?:\d(?:\.\d+)?|nano|flash|pro|ultra|live)",
            r"google\s+(?:gemini|bard|generative\s+ai)",
            r"bard\s+ai",
            r"palm[\s._-]?[12]",
        ),
    ),
    Rule(
        "named.meta_llama",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"(?:meta\s+)?llama[\s._-]?[2345](?:\.\d+)?",
            r"meta\s+ai",
            r"code\s*llama",
        ),
    ),
    Rule(
        "named.alibaba_qwen",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"qwen(?:[\s._-]?(?:\d(?:\.\d+)?|max|plus|turbo|coder|chat|vl|omni))?",
            r"tongyi\s+qianwen",
            r"通义千问",
        ),
    ),
    Rule(
        "named.deepseek",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"deep[\s._-]*seek(?:[\s._-]?(?:chat|coder|reasoner|r1|v\d|\d(?:\.\d+)?))?",
        ),
    ),
    Rule(
        "named.mistral_mixtral",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"mistral\s+ai",
            r"mistral[\s._-]?(?:\d(?:\.\d+)?|small|medium|large|nemo|codestral|ministral|magistral)",
            r"mixtral(?:[\s._-]?\d+)?",
            r"mistral(?:'s)?\s+le\s+chat",
            r"le\s+chat\s+by\s+mistral",
        ),
    ),
    Rule(
        "named.xai_grok",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"xai\s+grok",
            r"grok[\s._-]?(?:\d(?:\.\d+)?|ai|chat|assistant)",
        ),
    ),
    Rule(
        "named.cohere_command",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"cohere\s+(?:command|chat|generate|api|ai|model)",
            r"command[\s._-]r(?:[\s._-]?(?:plus|\+|7b|08-2024))?",
        ),
    ),
    Rule(
        "named.microsoft_copilot_phi",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"microsoft\s+(?:365\s+)?copilot",
            r"github\s+copilot",
            r"windows\s+copilot",
            r"phi[\s._-]?[2-9](?:\.\d+)?",
        ),
    ),
    Rule(
        "named.amazon_bedrock_nova_q",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"amazon\s+bedrock",
            r"aws\s+bedrock",
            r"amazon\s+q(?:\s+(?:developer|business))?",
            r"amazon\s+nova(?:\s+(?:micro|lite|pro|premier|sonic))?",
            r"titan\s+text(?:\s+(?:lite|express))?",
        ),
    ),
    Rule(
        "named.ai21_jamba",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"ai21(?:\s+labs)?",
            r"ai21\s+jamba",
            r"jamba\s+1\.5",
            r"jurassic[\s._-]?[12]",
        ),
    ),
    Rule(
        "named.chinese_llm_families",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"chat[\s._-]*glm",
            r"glm[\s._-]?[345](?:\.\d+)?",
            r"zhipu\s+ai",
            r"baichuan(?:[\s._-]?\d+)?",
            r"(?:01[\s._-]*ai\s+)?yi[\s._-]?(?:6|9|34|large|lightning)",
            r"kimi\s+(?:ai|chat|assistant|k\d(?:\.\d+)?)",
            r"moonshot\s+ai",
            r"doubao\s+(?:ai|llm|assistant|pro|lite)",
            r"ernie\s+(?:bot|\d(?:\.\d+)?)",
            r"wenxin\s+yiyan",
            r"sparkdesk",
            r"iflytek\s+spark",
            r"hunyuan(?:\s+llm)?",
            r"intern[\s._-]*lm(?:[\s._-]?\d+)?",
            r"minimax[\s._-](?:text|m\d|abab)",
            r"sensechat",
            r"sensenova",
            r"skywork[\s._-](?:llm|o\d)",
            r"xverse[\s._-]?(?:\d+)?",
        ),
    ),
    Rule(
        "named.open_model_families",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"gemma[\s._-]?[2-9](?:\.\d+)?",
            r"falcon[\s._-]?(?:7b|11b|40b|180b|mamba)",
            r"vicuna[\s._-]?(?:7b|13b|33b|\d+(?:\.\d+)?)",
            r"wizard[\s._-]*lm",
            r"nous[\s._-]*hermes",
            r"openhermes",
            r"zephyr[\s._-]?(?:7b|beta|alpha)",
            r"dbrx(?:\s+instruct)?",
            r"olmo(?:[\s._-]?\d+)?",
            r"nemotron(?:[\s._-]?\d+)?",
        ),
    ),
    Rule(
        "named.consumer_llm_services",
        "named_llm",
        "strong_identity",
        bounded_alternatives(
            r"perplexity\s+ai",
            r"poe\s+(?:ai|chatbot|assistant)",
            r"character\.ai",
            r"character\s+ai\s+(?:chat|chatbot|assistant|characters?|app)",
            r"hugging[\s._-]*chat",
            r"youchat",
            r"you\.com\s+(?:ai|chat)",
            r"phind(?:\s+ai)?",
            r"pi\s+(?:personal\s+)?ai",
            r"replika(?:\s+ai)?",
        ),
    ),
)


LLM_PLATFORM_RULES: tuple[Rule, ...] = (
    Rule(
        "platform.llm_cloud_and_aggregators",
        "llm_platform",
        "strong_identity",
        bounded_alternatives(
            r"openrouter",
            r"groq(?:cloud)?",
            r"together\s+ai",
            r"fireworks\s+ai",
            r"cerebras\s+inference",
            r"sambanova(?:\s+cloud)?",
        ),
    ),
    Rule(
        "platform.llm_orchestration_runtime",
        "llm_platform",
        "strong_identity",
        bounded_alternatives(
            r"lang[\s._-]*chain(?:4j)?",
            r"llama[\s._-]*index",
            r"semantic\s+kernel",
            r"ollama",
            r"lm\s+studio",
            r"vllm",
            r"llama\.cpp",
            r"mlc[\s._-]*llm",
            r"tensor[\s._-]*rt[\s._-]*llm",
            r"media[\s._-]*pipe\s+(?:tasks\s+)?(?:genai|llm\s+inference)",
            r"android\s+ai\s*core",
            r"executorch\s+llm",
            r"onnx\s+runtime\s+genai",
            r"lite[\s._-]*rt[\s._-]*lm",
            r"anything[\s._-]*llm",
            r"gpt[\s._-]*4[\s._-]*all",
            r"open[\s._-]*webui",
            r"libre[\s._-]*chat",
            r"local[\s._-]*ai",
            r"jan\s+ai",
            r"dify(?:\.ai)?",
            r"flowise(?:\s+ai)?",
            r"firebase\s+genkit",
        ),
    ),
)


LLM_TECHNOLOGY_RULES: tuple[Rule, ...] = (
    Rule(
        "technology.language_models",
        "llm_technology",
        "explicit_llm_technology",
        bounded_alternatives(
            r"large\s+language\s+models?",
            r"\bLLMs?\b",
            r"small\s+language\s+models?",
            r"generative\s+language\s+models?",
            r"foundation\s+language\s+models?",
            r"multimodal\s+language\s+models?",
            r"vision[\s-]+language\s+models?",
            r"language[\s-]+capable\s+multimodal\s+models?",
            r"natural\s+language\s+generation",
        ),
    ),
    Rule(
        "technology.rag_agents_protocols",
        "llm_technology",
        "explicit_llm_technology",
        bounded_alternatives(
            r"retrieval[\s-]+augmented\s+generation",
            r"model\s+context\s+protocol",
            r"agentic\s+(?:ai|workflow|assistant|system)",
            r"autonomous\s+ai\s+agents?",
            r"multi[\s-]+agent\s+(?:ai|system|workflow)",
            r"(?:llm|language\s+model)\s+(?:agents?|function\s+calling|tool\s+calling)",
            r"(?:function|tool)\s+calling\s+(?:model|assistant|agent|api)",
        ),
    ),
    Rule(
        "technology.prompts_and_inference",
        "llm_technology",
        "explicit_llm_technology",
        bounded_alternatives(
            r"prompt\s+engineering",
            r"system\s+prompts?",
            r"chat\s+completions?\s+api",
            r"text\s+completions?\s+api",
            r"llm\s+inference",
            r"on[\s-]+device\s+(?:llm|language\s+model)",
            r"local\s+(?:llm|language\s+model)",
            r"gguf\s+(?:model|file|format)",
        ),
    ),
)


AI_FEATURE_PREFIX = (
    r"(?:ai|a\.i\.|artificial\s+intelligence|generative\s+ai|gen[\s-]*ai|gpt)"
    r"(?:[\s-]+(?:powered|driven|enabled|enhanced|assisted|based|generated))?"
    r"[\s:–—-]*(?:(?:virtual|personal|intelligent|smart|conversational)[\s-]+)?"
)
CHAT_ROLE_PATTERN = (
    r"(?:chat[\s-]*bots?|chat|assistants?|copilots?|agents?|companions?|friends?|"
    r"girlfriends?|boyfriends?|tutors?|coaches?|advis(?:or|er)s?|role[\s-]*play(?:ing)?)"
)
TEXT_FEATURE_PATTERN = (
    r"(?:writers?|writing(?:\s+assistants?)?|copywriters?|"
    r"(?:text|content|story|essay|email|caption|code)\s+generat(?:ors?|ion)|"
    r"(?:essay|email|story)\s+writers?|coding\s+assistants?|"
    r"summari[sz](?:e[rs]?|ing|ations?)|summar(?:y|ies)|"
    r"paraphras(?:e[rs]?|ing)|rewrit(?:e[rs]?|ing)|replies|responses|answers)"
)

LANGUAGE_FUNCTION_RULES: tuple[Rule, ...] = (
    Rule(
        "function.ai_chat_assistant_agent",
        "language_function",
        "explicit_language_function",
        bounded_alternatives(
            AI_FEATURE_PREFIX + CHAT_ROLE_PATTERN,
            CHAT_ROLE_PATTERN + r"\s+(?:is\s+)?(?:powered\s+by|using|with|based\s+on)\s+(?:ai|artificial\s+intelligence|generative\s+ai|gpt)",
            r"(?:ai|a\.i\.|artificial\s+intelligence|generative\s+ai|gen\s*ai)[\s-]*(?:chat\s*bot|chatbot|assistant|copilot|agent|companion|tutor|coach|advisor)",
            r"(?:chat\s*bot|chatbot|assistant|copilot|agent|companion|tutor|coach|advisor)\s+(?:powered\s+by|using|with|based\s+on)\s+(?:ai|artificial\s+intelligence|generative\s+ai)",
            r"chat\s+with\s+(?:(?:an?|your|the)\s+)?ai",
            r"(?:ask|talk|speak)\s+(?:to\s+)?(?:the\s+)?ai",
            r"conversational\s+(?:ai|assistant|agent)",
            r"virtual\s+ai\s+assistant",
            r"personal\s+ai\s+assistant",
            r"ai\s+answer\s+engine",
            r"ai\s+search\s+engine",
        ),
    ),
    Rule(
        "function.chat_over_content",
        "language_function",
        "explicit_language_function",
        bounded_alternatives(
            r"chat\s+with\s+(?:your\s+|any\s+|a\s+)?(?:pdf|documents?|websites?|web\s*pages?|books?|notes?|data|database|knowledge\s+base)",
            r"ask\s+(?:questions?\s+)?(?:about|of)\s+(?:your\s+|any\s+|a\s+)?(?:pdf|documents?|files?|websites?|images?|books?|notes?|data|knowledge\s+base)",
            r"document\s+(?:chat|question[\s-]*answering|q\s*&\s*a)",
            r"(?:pdf|document|website)\s+(?:chat|assistant|copilot)",
        ),
    ),
    Rule(
        "function.ai_writing_text_code",
        "language_function",
        "explicit_language_function",
        bounded_alternatives(
            AI_FEATURE_PREFIX + TEXT_FEATURE_PATTERN,
            TEXT_FEATURE_PATTERN + r"\s+(?:is\s+)?(?:powered\s+by|using|with|based\s+on)\s+(?:ai|artificial\s+intelligence|generative\s+ai|gpt)",
            r"(?:ai|gpt)[\s-]+(?:generated|written|drafted)[\s-]+(?:texts?|emails?|essays?|stories|articles?|blogs?|captions?|code|scripts?|poems?|lyrics|reports?|replies|responses|answers)",
            r"(?:generat(?:e[sd]?|ing)|write[sn]?|writing|draft(?:s|ed|ing)?|rewrit(?:e[sn]?|ing)|summari[sz](?:e[sd]?|ing)|paraphras(?:e[sd]?|ing)|explain(?:s|ed|ing)?|answer(?:s|ed|ing)?)\s+(?:using|with|through|via)\s+(?:ai|artificial\s+intelligence|generative\s+ai|gpt)",
            r"ai[\s-]*(?:writer|writing\s+assistant|text\s+generator|content\s+generator|story\s+generator|essay\s+writer|email\s+writer|copywriter|code\s+generator|coding\s+assistant|summari[sz]er|paraphraser|rewriter)",
            r"(?:writer|writing\s+assistant|text\s+generator|content\s+generator|story\s+generator|essay\s+writer|email\s+writer|copywriter|code\s+generator|coding\s+assistant|summari[sz]er|paraphraser|rewriter)\s+(?:powered\s+by|using|with)\s+(?:ai|artificial\s+intelligence)",
            r"(?:generate|write|draft|rewrite|summari[sz]e|paraphrase|explain|answer)\s+(?:using|with|through|via)\s+(?:ai|artificial\s+intelligence|generative\s+ai)",
        ),
    ),
)


GENERIC_GENAI_RULES: tuple[Rule, ...] = (
    Rule(
        "generic.generative_ai",
        "generic_genai",
        "generic_genai_ambiguous",
        bounded_alternatives(
            r"generative\s+ai",
            r"gen[\s-]*ai",
            r"aigc",
            r"generative\s+artificial\s+intelligence",
            r"generative\s+models?",
            r"foundation\s+models?",
            r"multimodal\s+ai",
        ),
    ),
)


NONLANGUAGE_GENAI_RULES: tuple[Rule, ...] = (
    Rule(
        "nonlanguage.image_generation",
        "nonlanguage_genai",
        "nonlanguage_control",
        bounded_alternatives(
            r"text[\s-]+to[\s-]+image",
            r"ai\s+(?:image|photo|art|avatar|logo)\s+(?:generator|generation|creator|maker)",
            r"ai\s+(?:image|photo|art(?:work)?|anime|avatar|logo)\b",
            r"(?:image|photo|art|avatar|logo)\s+(?:generator|generation|creator|maker)\s+(?:powered\s+by|using|with)?\s*ai",
            r"(?:image|photo|art(?:work)?|anime|avatar|logo)\s+ai\s+(?:generator|generation|creator|maker)?",
            r"stable\s+diffusion",
            r"midjourney",
            r"dall[\s._-]*e(?:\s*\d+)?",
            r"dreambooth",
            r"controlnet",
            r"(?:flux|imagen)[\s._-]?(?:ai|image|\d(?:\.\d+)?)",
            r"background\s+remov(?:al|er)\s+(?:with\s+)?ai",
            r"ai\s+(?:upscaler|face\s+swap|photo\s+editor)",
            r"turn\s+(?:your\s+)?(?:photos?|pictures?|text|words?)\s+into\s+(?:[^.]{0,50}\s+)?(?:images?|art|anime|avatars?)",
        ),
    ),
    Rule(
        "nonlanguage.video_generation",
        "nonlanguage_genai",
        "nonlanguage_control",
        bounded_alternatives(
            r"text[\s-]+to[\s-]+video",
            r"ai\s+video\s+(?:generator|generation|creator|maker)",
            r"ai\s+(?:generated\s+)?videos?\b",
            r"video\s+(?:generator|generation|creator|maker)\s+(?:powered\s+by|using|with)?\s*ai",
            r"runway\s+(?:gen[\s._-]?\d|ai\s+video)",
            r"sora\s+(?:ai|video|openai)",
            r"veo[\s._-]?(?:ai|video|\d)",
            r"kling\s+ai",
            r"pika\s+(?:labs|ai|video)",
        ),
    ),
    Rule(
        "nonlanguage.audio_speech_music",
        "nonlanguage_genai",
        "nonlanguage_control",
        bounded_alternatives(
            r"text[\s-]+to[\s-]+speech",
            r"voice\s+clon(?:e|ing)",
            r"ai\s+(?:voice|speech|music|song|audio)\s+(?:generator|generation|creator|maker)",
            r"ai\s+(?:generated\s+)?(?:voice|speech|music|songs?|audio)\b",
            r"(?:music|song|voice|audio)\s+(?:generator|generation|creator|maker)\s+(?:powered\s+by|using|with)?\s*ai",
            r"suno\s+(?:ai|music)",
            r"udio\s+(?:ai|music)",
            r"musicgen",
            r"whisper\s+(?:speech|transcription|transcriber|model|api)",
            r"ai\s+(?:transcription|transcriber|speech\s+recognition)",
        ),
    ),
)


TRADITIONAL_AI_RULES: tuple[Rule, ...] = (
    Rule(
        "traditional.machine_deep_learning",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"machine\s+learning",
            r"deep\s+learning",
            r"neural\s+networks?",
            r"supervised\s+learning",
            r"unsupervised\s+learning",
            r"random\s+forests?",
            r"support\s+vector\s+machines?",
            r"decision\s+trees?",
            r"naive\s+bayes",
            r"k[\s-]*nearest\s+neighbou?rs?",
        ),
    ),
    Rule(
        "traditional.cv_ocr",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"computer\s+vision",
            r"optical\s+character\s+recognition",
            r"ocr",
            r"object\s+(?:detection|recognition|tracking)",
            r"face\s+(?:detection|recognition|verification)",
            r"facial\s+(?:detection|recognition|verification)",
            r"image\s+(?:classification|segmentation|recognition)",
            r"pose\s+estimation",
        ),
    ),
    Rule(
        "traditional.speech_audio",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"automatic\s+speech\s+recognition",
            r"speech\s+recognition",
            r"voice\s+recognition",
            r"speaker\s+(?:recognition|identification|verification)",
            r"speech[\s-]+to[\s-]+text",
            r"text[\s-]+to[\s-]+speech",
            r"audio\s+transcription",
        ),
    ),
    Rule(
        "traditional.nlp",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"machine\s+translation",
            r"spell\s+check(?:er|ing)?",
            r"grammar\s+check(?:er|ing)?",
            r"sentiment\s+analysis",
            r"named\s+entity\s+recognition",
            r"text\s+classification",
            r"keyword\s+extraction",
            r"part[\s-]+of[\s-]+speech\s+tagging",
            r"stemming\s+and\s+lemmati[sz]ation",
        ),
    ),
    Rule(
        "traditional.prediction_recommendation",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"recommendation\s+(?:system|engine|algorithm)",
            r"predictive\s+analytics",
            r"anomaly\s+detection",
            r"time[\s-]+series\s+forecasting",
            r"classification\s+model",
            r"prediction\s+model",
        ),
    ),
    Rule(
        "traditional.rule_template_chat",
        "traditional_ai",
        "traditional_non_llm",
        bounded_alternatives(
            r"rule[\s-]+based\s+(?:chat\s*bot|assistant|system)",
            r"scripted\s+(?:chat\s*bot|assistant|responses?)",
            r"template[\s-]+based\s+(?:chat\s*bot|assistant|responses?)",
            r"keyword[\s-]+based\s+(?:chat\s*bot|assistant|responses?)",
            r"predefined\s+(?:answers?|responses?|replies)",
            r"faq\s+(?:chat\s*bot|bot|assistant)",
            r"automated\s+replies\s+to\s+frequently\s+asked\s+questions",
        ),
    ),
)


AMBIGUITY_RULES: tuple[Rule, ...] = (
    Rule(
        "ambiguity.game_ai",
        "obvious_ambiguity",
        "excluded_ambiguity",
        bounded_alternatives(
            r"ai\s+(?:opponent|enemy|player|difficulty|controlled|bot)\b",
            r"(?:opponent|enemy|npc|computer)\s+ai\b",
            r"play\s+(?:against|versus|vs\.?)\s+(?:the\s+)?ai\b",
            r"game\s+ai\b",
        ),
    ),
    Rule(
        "ambiguity.ai_file",
        "obvious_ambiguity",
        "excluded_ambiguity",
        bounded_alternatives(
            r"\.ai\s+files?",
            r"adobe\s+illustrator",
            r"illustrator\s+files?",
        ),
    ),
    Rule(
        "ambiguity.artificial_insemination",
        "obvious_ambiguity",
        "excluded_ambiguity",
        bounded_alternatives(r"artificial\s+insemination"),
    ),
)


DIRECT_RULES = (
    NAMED_LLM_RULES
    + LLM_PLATFORM_RULES
    + LLM_TECHNOLOGY_RULES
    + LANGUAGE_FUNCTION_RULES
    + GENERIC_GENAI_RULES
    + NONLANGUAGE_GENAI_RULES
    + TRADITIONAL_AI_RULES
    + AMBIGUITY_RULES
)

AI_CONTEXT_RE = bounded_alternatives(
    r"ai",
    r"a\.i\.",
    r"artificial\s+intelligence",
    r"generative\s+ai",
    r"gen[\s-]*ai",
    r"aigc",
    r"language\s+model",
    r"foundation\s+model",
)

LANGUAGE_CAPABILITY_RE = bounded_alternatives(
    r"chat(?:[\s-]*bots?|ting)?",
    r"convers(?:e|ations?|ational)",
    r"dialog(?:ue)?s?",
    r"assistants?",
    r"copilots?",
    r"companions?|girlfriends?|boyfriends?|role[\s-]*play(?:ing)?",
    r"agents?",
    r"ask\s+(?:anything|questions?)",
    r"answers?\s+(?:(?:your|any)\s+)?(?:anything|questions?)",
    r"question[\s-]+answering",
    r"writ(?:ers?|ing)",
    r"write\s+(?:text|emails?|essays?|stories|articles?|blogs?|captions?|resumes?|cover\s+letters?|code|scripts?|poems?|lyrics|reports?)",
    r"draft(?:ing)?",
    r"rewrit(?:e|er|ing)",
    r"summari[sz](?:e[rs]?|ing|ations?)",
    r"summar(?:y|ies)",
    r"paraphras(?:e[rs]?|ing)",
    r"generat(?:e[sd]?|ing)\s+(?:texts?|emails?|essays?|stories|articles?|blogs?|captions?|resumes?|cover\s+letters?|code|scripts?|poems?|lyrics|reports?|replies|responses|answers)",
    r"reasoning",
    r"knowledge\s+base",
    r"document\s+(?:analysis|understanding)",
)

# Cross-field evidence needs a concrete language function, not merely a generic
# "assistant", social chat, a URL, or an unrelated model/API word.
STRONG_LANGUAGE_CAPABILITY_RE = bounded_alternatives(
    r"chat[\s-]*bots?",
    r"(?:chat|conversational|writing|coding|language)\s+assistants?",
    r"(?:free[\s-]*form|open[\s-]*ended)\s+(?:chat|conversations?|responses?|answers?)",
    r"summari[sz](?:e[rs]?|ing|ations?)|summar(?:y|ies)",
    r"paraphras(?:e[rs]?|ing)|rewrit(?:e[rs]?|ing)",
    r"(?:text|content|story|essay|email|caption|code)\s+generat(?:ors?|ion)",
    r"(?:generat(?:e[sd]?|ing)|write[sn]?|writing|draft(?:s|ed|ing)?)\s+(?:texts?|emails?|essays?|stories|articles?|blogs?|captions?|code|scripts?|poems?|lyrics|reports?|replies|responses|answers)",
    r"chat\s+with\s+(?:your\s+|any\s+|a\s+)?(?:pdf|documents?|files?|books?|notes?|knowledge\s+base)",
    r"document\s+(?:chat|question[\s-]*answering|q\s*&\s*a)",
)
RULE_ONLY_CHAT_RE = bounded_alternatives(
    r"(?:chat[\s-]*bots?|assistants?)\s+(?:(?:that|which|is|are)\s+)?(?:only|solely|entirely|strictly)\s+(?:uses?\s+)?(?:rules?|scripts?|templates?|predefined\s+(?:answers?|responses?|replies))",
    r"(?:chat[\s-]*bots?|assistants?)\s+uses?\s+(?:only|solely)\s+(?:rules?|scripts?|templates?|predefined\s+(?:answers?|responses?|replies))",
    r"(?:rule|script|template)[\s-]*only\s+(?:(?:ai|intelligent)\s+)?(?:chat[\s-]*bots?|assistants?)",
    r"(?:chat[\s-]*bots?|assistants?)\s+(?:is\s+|are\s+)?(?:only|solely|entirely|strictly)\s+(?:rule|script|template)[\s-]*based",
    r"(?:rule|script|template)[\s-]*based\s+(?:chat[\s-]*bots?|assistants?)\s+only",
)
RULE_ONLY_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|rather\s+than|beyond)(?:\s+(?:an?|the))?\s*$", re.IGNORECASE
)

CONTEXTUAL_MODEL_RE = bounded_alternatives(
    r"claude",
    r"gemini",
    r"bard",
    r"llama",
    r"mistral",
    r"gemma",
    r"grok",
    r"phi",
    r"falcon",
    r"nova",
    r"titan",
    r"copilot",
    r"perplexity",
    r"poe",
    r"kimi",
    r"doubao",
    r"ernie",
    r"jamba",
    r"yi",
    r"rag",
    r"vlm",
    r"slm",
    r"o[134]",
)

MODEL_CONTEXT_RE = bounded_alternatives(
    r"ai",
    r"artificial\s+intelligence",
    r"language\s+model",
    r"chat[\s-]*bots?",
    r"assistants?",
    r"chat(?:ting)?",
    r"summar(?:y|ies)|summari[sz](?:e[rs]?|ing|ations?)",
    r"prompt",
    r"generat(?:e|ive|ion)",
    r"reasoning",
    r"api",
    r"model",
    r"inference",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def ruleset_sha256() -> str:
    payload = [
        {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "tier": rule.tier,
            "pattern": rule.pattern.pattern,
            "flags": rule.pattern.flags,
        }
        for rule in DIRECT_RULES
    ]
    payload.extend(
        [
            {"rule_id": "context.ai_language", "pattern": AI_CONTEXT_RE.pattern},
            {"rule_id": "context.language_capability", "pattern": LANGUAGE_CAPABILITY_RE.pattern},
            {"rule_id": "context.model_family", "pattern": CONTEXTUAL_MODEL_RE.pattern},
            {"rule_id": "context.model_guard", "pattern": MODEL_CONTEXT_RE.pattern},
            {"rule_id": "context.cross_field_language", "pattern": STRONG_LANGUAGE_CAPABILITY_RE.pattern},
            {"rule_id": "policy.rule_only_chat", "pattern": RULE_ONLY_CHAT_RE.pattern},
            {"rule_id": "policy.rule_only_negation", "pattern": RULE_ONLY_NEGATION_RE.pattern},
            {
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "context_window_chars": CONTEXT_WINDOW_CHARS,
                "cross_field_evidence_enabled": True,
            },
        ]
    )
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def evidence_snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip()


def direct_matches(
    fields: Iterable[tuple[str, str]], max_per_rule: int = 3
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for source_field, text in fields:
        if not text:
            continue
        for rule in DIRECT_RULES:
            for index, match in enumerate(rule.pattern.finditer(text)):
                if index >= max_per_rule:
                    break
                matches.append(
                    {
                        "rule_id": rule.rule_id,
                        "category": rule.category,
                        "tier": rule.tier,
                        "source_field": source_field,
                        "match": match.group(0),
                        "snippet": evidence_snippet(text, match.start(), match.end()),
                    }
                )
    return matches


def span_gap(left: re.Match[str], right: re.Match[str]) -> int:
    if left.end() < right.start():
        return right.start() - left.end()
    if right.end() < left.start():
        return left.start() - right.end()
    return 0


def closest_cooccurrence(
    text: str,
    left_pattern: re.Pattern[str],
    right_pattern: re.Pattern[str],
) -> tuple[int, int] | None:
    left_matches = list(left_pattern.finditer(text))
    right_matches = list(right_pattern.finditer(text))
    best: tuple[int, int, int] | None = None
    left_index = right_index = 0
    # Sorted, non-overlapping matches allow a linear sweep. Do not silently
    # ignore language features merely because they occur after match 40.
    while left_index < len(left_matches) and right_index < len(right_matches):
        left, right = left_matches[left_index], right_matches[right_index]
        gap = span_gap(left, right)
        if gap <= CONTEXT_WINDOW_CHARS:
            candidate = (gap, min(left.start(), right.start()), max(left.end(), right.end()))
            if best is None or candidate < best:
                best = candidate
        if left.end() <= right.end():
            left_index += 1
        else:
            right_index += 1
    return None if best is None else (best[1], best[2])


def contextual_matches(fields: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    scanned_fields: list[tuple[str, str, str]] = []
    for source_field, text in fields:
        if not text:
            continue
        # URLs and email addresses often contain tokens such as "chat" but are
        # not descriptions of an app capability.  Replace them with equal-size
        # whitespace so evidence offsets still refer to the normalized text.
        scan_text = URL_OR_EMAIL_RE.sub(lambda match: " " * len(match.group(0)), text)
        scanned_fields.append((source_field, text, scan_text))
        ai_language = closest_cooccurrence(
            scan_text, AI_CONTEXT_RE, LANGUAGE_CAPABILITY_RE
        )
        if ai_language:
            start, end = ai_language
            matches.append(
                {
                    "rule_id": "context.ai_plus_language_capability",
                    "category": "contextual_language_function",
                    "tier": "contextual_language_function",
                    "source_field": source_field,
                    "match": text[start:end],
                    "snippet": evidence_snippet(text, start, end),
                }
            )

        model_context = closest_cooccurrence(
            scan_text, CONTEXTUAL_MODEL_RE, MODEL_CONTEXT_RE
        )
        if model_context:
            start, end = model_context
            matches.append(
                {
                    "rule_id": "context.model_family_plus_ai_language",
                    "category": "contextual_model_identity",
                    "tier": "contextual_model_identity",
                    "source_field": source_field,
                    "match": text[start:end],
                    "snippet": evidence_snippet(text, start, end),
                }
            )
    has_same_field_ai_language = any(
        match["rule_id"] == "context.ai_plus_language_capability" for match in matches
    )
    if not has_same_field_ai_language:
        for left_field, left_text, left_scan in scanned_fields:
            ai_match = AI_CONTEXT_RE.search(left_scan)
            if not ai_match:
                continue
            for right_field, right_text, right_scan in scanned_fields:
                if left_field == right_field:
                    continue
                language_match = STRONG_LANGUAGE_CAPABILITY_RE.search(right_scan)
                if not language_match:
                    continue
                evidence = [
                    {"source_field": left_field, "match": ai_match.group(0),
                     "snippet": evidence_snippet(left_text, ai_match.start(), ai_match.end())},
                    {"source_field": right_field, "match": language_match.group(0),
                     "snippet": evidence_snippet(right_text, language_match.start(), language_match.end())},
                ]
                matches.append({
                    "rule_id": "context.ai_language_across_fields",
                    "category": "contextual_language_function",
                    "tier": "contextual_language_function",
                    "source_field": "description+recentChanges",
                    "match": " + ".join(item["match"] for item in evidence),
                    "snippet": " | ".join(item["source_field"] + ": " + item["snippet"] for item in evidence),
                    "source_evidence": evidence,
                })
                break
            if any(match["rule_id"] == "context.ai_language_across_fields" for match in matches):
                break
    return matches


PRIMARY_CATEGORY_ORDER = (
    "named_llm",
    "llm_platform",
    "llm_technology",
    "language_function",
    "contextual_model_identity",
    "contextual_language_function",
)

PRIMARY_TIER_BY_CATEGORY = {
    "named_llm": "strong_identity",
    "llm_platform": "strong_identity",
    "llm_technology": "explicit_llm_technology",
    "language_function": "explicit_language_function",
    "contextual_model_identity": "contextual_model_identity",
    "contextual_language_function": "contextual_language_function",
}


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    description = normalize_text(record.get("description"))
    recent_changes = normalize_text(record.get("recentChanges"))
    fields = (("description", description), ("recentChanges", recent_changes))
    matches = direct_matches(fields)
    matches.extend(contextual_matches(fields))

    by_category: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        by_category.setdefault(match["category"], []).append(match)

    candidate_tier: str | None = None
    for category in PRIMARY_CATEGORY_ORDER:
        if by_category.get(category):
            candidate_tier = PRIMARY_TIER_BY_CATEGORY[category]
            break

    generic_matches = by_category.get("generic_genai", [])
    nonlanguage_matches = by_category.get("nonlanguage_genai", [])
    traditional_matches = by_category.get("traditional_ai", [])
    ambiguity_matches = by_category.get("obvious_ambiguity", [])

    # A generic GenAI claim without an explicit non-language modality remains
    # in the primary pool to protect recall.  Generic GenAI attached only to
    # image/video/audio functionality goes to the separate control pool.
    if candidate_tier is None and generic_matches and not nonlanguage_matches and not ambiguity_matches:
        candidate_tier = "generic_genai_ambiguous"

    matched_rule_ids = {match["rule_id"] for match in matches}
    has_explicit_llm_identity = bool(
        by_category.get("named_llm")
        or by_category.get("llm_platform")
        or by_category.get("llm_technology")
    )
    has_strong_language_generation = bool(
        matched_rule_ids
        & {"function.ai_writing_text_code", "function.chat_over_content"}
    )
    has_rule_only_chat = any(
        not RULE_ONLY_NEGATION_RE.search(text[max(0, match.start() - 40):match.start()])
        for _, text in fields for match in RULE_ONLY_CHAT_RE.finditer(text)
    )
    excluded_reason: str | None = None
    # Traditional terminology is not evidence that a separate language-AI
    # feature is absent. Only an explicitly exclusive rule/template chat claim
    # can reject a weak chat candidate; actual generation or LLM evidence wins.
    if (
        candidate_tier is not None and has_rule_only_chat
        and not has_explicit_llm_identity and not has_strong_language_generation
    ):
        excluded_reason = "explicitly_rule_only_chat_without_generation_evidence"
        candidate_tier = None

    is_llm_candidate = candidate_tier is not None
    is_nonlanguage_genai_control = (
        not is_llm_candidate and bool(nonlanguage_matches)
    )

    selected_matches = [
        match
        for match in matches
        if match["category"]
        not in {"nonlanguage_genai", "traditional_ai", "obvious_ambiguity"}
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "ruleset_version": RULESET_VERSION,
        "ruleset_sha256": ruleset_sha256(),
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "is_llm_candidate": is_llm_candidate,
        "candidate_tier": candidate_tier,
        "excluded_reason": excluded_reason,
        "traditional_ai_coexists_with_llm_signal": is_llm_candidate and bool(traditional_matches),
        "cross_field_evidence": "context.ai_language_across_fields" in matched_rule_ids,
        "is_nonlanguage_genai_control": is_nonlanguage_genai_control,
        "matched_categories": sorted(
            category
            for category in by_category
            if category
            not in {"nonlanguage_genai", "traditional_ai", "obvious_ambiguity"}
        ),
        "matched_rules": selected_matches,
        "nonlanguage_genai_matches": nonlanguage_matches,
        "traditional_ai_matches": traditional_matches,
        "ambiguity_matches": ambiguity_matches,
        "normalized_text_sha256": sha256_text(f"{description}\0{recent_changes}"),
    }


def output_record(record: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    enriched["llm_keyword_analysis"] = analysis
    return enriched


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def temp_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def validate_paths(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise ValueError(f"input file does not exist: {args.input}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    paths = [
        args.input.resolve(),
        args.output.resolve(),
        args.genai_control_output.resolve(),
        args.summary_output.resolve(),
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("input and output paths must all be distinct")
    if not args.force:
        existing = [str(path) for path in paths[1:] if path.exists()]
        if existing:
            raise ValueError(
                "refusing to replace existing output(s) without --force: "
                + ", ".join(existing)
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    parser = argparse.ArgumentParser(
        description="Retrieve LLM-scoped app candidates from Play text"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=data_dir / "extracted_text_fields_english.jsonl",
        help="Translated/cleaned JSONL with docid, description, and recentChanges",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / "google_play_apps_llm_keyword_candidates.jsonl",
        help="Primary LLM-oriented candidate JSONL",
    )
    parser.add_argument(
        "--genai-control-output",
        type=Path,
        default=data_dir / "google_play_apps_nonlanguage_genai_controls.jsonl",
        help="Pure image/video/audio GenAI control-pool JSONL",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=data_dir / "google_play_apps_llm_keyword_summary.json",
        help="Run manifest and rule-level counts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many valid records for a pilot run",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print progress after this many scanned non-empty lines",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace outputs if they already exist",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_paths(args)
    started_at = utc_now()
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    control_digest = hashlib.sha256()
    seen_docids: set[str] = set()
    counts = Counter()
    tier_counts = Counter()
    category_counts = Counter()
    rule_counts = Counter()
    reached_eof = True

    for path in (args.output, args.genai_control_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_temp = temp_output_path(args.output)
    control_temp = temp_output_path(args.genai_control_output)

    with args.input.open("rb") as input_handle, \
            output_temp.open("w", encoding="utf-8", newline="\n") as output_handle, \
            control_temp.open("w", encoding="utf-8", newline="\n") as control_handle:
        for line_number, raw_line in enumerate(input_handle, 1):
            if args.limit is not None and counts["valid_unique_records"] >= args.limit:
                reached_eof = False
                break
            input_digest.update(raw_line)
            if not raw_line.strip():
                counts["empty_lines"] += 1
                continue
            counts["nonempty_lines"] += 1
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["malformed_lines"] += 1
                continue
            if not isinstance(record, dict):
                counts["non_object_lines"] += 1
                continue
            docid = record.get("docid")
            if not isinstance(docid, str) or not docid.strip():
                counts["missing_docids"] += 1
                continue
            docid = docid.strip()
            if docid in seen_docids:
                counts["duplicate_docids"] += 1
                continue
            seen_docids.add(docid)
            record["docid"] = docid
            counts["valid_unique_records"] += 1

            analysis = classify_record(record)
            present_categories = set(analysis["matched_categories"])
            present_categories.update(
                match["category"] for match in analysis["nonlanguage_genai_matches"]
            )
            present_categories.update(
                match["category"] for match in analysis["traditional_ai_matches"]
            )
            present_categories.update(
                match["category"] for match in analysis["ambiguity_matches"]
            )
            category_counts.update(present_categories)
            present_rules = {
                match["rule_id"]
                for key in (
                    "matched_rules",
                    "nonlanguage_genai_matches",
                    "traditional_ai_matches",
                    "ambiguity_matches",
                )
                for match in analysis[key]
            }
            rule_counts.update(present_rules)

            if analysis["excluded_reason"]:
                counts["traditional_ai_exclusions"] += 1

            if analysis["is_llm_candidate"]:
                if analysis["traditional_ai_coexists_with_llm_signal"]:
                    counts["traditional_ai_coexisting_candidates"] += 1
                if analysis["cross_field_evidence"]:
                    counts["cross_field_candidates"] += 1
                encoded = (json.dumps(output_record(record, analysis), ensure_ascii=False) + "\n").encode("utf-8")
                output_handle.write(encoded.decode("utf-8"))
                output_digest.update(encoded)
                counts["llm_candidates"] += 1
                tier_counts[analysis["candidate_tier"]] += 1
            elif analysis["is_nonlanguage_genai_control"]:
                encoded = (json.dumps(output_record(record, analysis), ensure_ascii=False) + "\n").encode("utf-8")
                control_handle.write(encoded.decode("utf-8"))
                control_digest.update(encoded)
                counts["nonlanguage_genai_controls"] += 1
            else:
                counts["not_selected"] += 1

            if counts["nonempty_lines"] % args.progress_every == 0:
                print(
                    f"Scanned {counts['nonempty_lines']:,} lines | "
                    f"LLM candidates={counts['llm_candidates']:,} | "
                    f"non-language GenAI controls={counts['nonlanguage_genai_controls']:,}",
                    flush=True,
                )

    os.replace(output_temp, args.output)
    os.replace(control_temp, args.genai_control_output)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "ruleset_version": RULESET_VERSION,
        "ruleset_sha256": ruleset_sha256(),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "complete_input_scan": reached_eof,
        "limit": args.limit,
        "context_window_chars": CONTEXT_WINDOW_CHARS,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "cross_field_evidence_enabled": True,
        "method_note": (
            "Keyword retrieval is an LLM-oriented candidate stage, not an LLM label. "
            "Traditional ML/DL, OCR, speech, translation and pure non-language AI "
            "signals alone do not qualify an app. Actual AI language-feature signals "
            "are retained even when traditional terminology coexists. Only explicitly "
            "exclusive rule-only chat without generation/LLM evidence is rejected. "
            "Strong language clues may be linked across description/recent changes; "
            "coexisting/cross-field counts are nonexclusive subsets. Candidates require "
            "semantic screening and APK evidence."
        ),
        "input": {
            "path": str(args.input.resolve()),
            "size_bytes": args.input.stat().st_size,
            "scanned_prefix_sha256": input_digest.hexdigest(),
            "sha256": input_digest.hexdigest() if reached_eof else None,
        },
        "counts": dict(sorted(counts.items())),
        "candidate_tiers": dict(sorted(tier_counts.items())),
        "matched_categories_by_record": dict(sorted(category_counts.items())),
        "matched_rules_by_record": dict(sorted(rule_counts.items())),
        "outputs": {
            "llm_candidates": {
                "path": str(args.output.resolve()),
                "sha256": output_digest.hexdigest(),
                "records": counts["llm_candidates"],
            },
            "nonlanguage_genai_controls": {
                "path": str(args.genai_control_output.resolve()),
                "sha256": control_digest.hexdigest(),
                "records": counts["nonlanguage_genai_controls"],
            },
        },
    }
    atomic_write_json(args.summary_output, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except KeyboardInterrupt:
        print("Interrupted; temporary output files were retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    counts = summary["counts"]
    print(
        "Completed keyword retrieval: "
        f"scanned={counts.get('valid_unique_records', 0):,}, "
        f"llm_candidates={counts.get('llm_candidates', 0):,}, "
        f"nonlanguage_genai_controls={counts.get('nonlanguage_genai_controls', 0):,}"
    )
    print(f"LLM candidates: {args.output.resolve()}")
    print(f"GenAI controls: {args.genai_control_output.resolve()}")
    print(f"Summary: {args.summary_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
