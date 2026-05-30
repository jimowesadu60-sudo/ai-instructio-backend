"""API1: user intake → result1."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from api.ai_client import AIClient
from utils.json_parser import JSONParseError, parse_json_from_llm, validate_json
from utils.prompt_loader import load_prompt

JSON_OBJECT_FORMAT: Dict[str, str] = {"type": "json_object"}

USER_FOOTER = (
    "以上为完整任务说明与占位符已替换后的输入。"
    "请严格只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释性文字。"
)

# Multimodal path uses a vision-capable model (per product requirement).
_VISION_MODEL = "gpt-4o-mini"

_RESULT1_INNER_KEYS: List[str] = [
    "schema_version",
    "source_trace",
    "input_digest",
    "core_decision",
    "theme_analysis",
    "goal_semantics",
    "condition_semantics",
    "platform_adaptation",
    "audience_focus",
    "presentation_decision",
    "narration_decision",
    "material_analysis",
    "media_style_inference",
    "execution_guidance",
    "risk_and_confirmation",
    "editable_fields",
]

_MATERIAL_ANALYSIS_KEYS: List[str] = [
    "material_status",
    "material_count_estimate",
    "material_types",
    "available_elements",
    "usable_shot_directions",
    "supported_presentation_modes",
    "supported_narration_modes",
    "supported_content_directions",
    "material_risks",
    "material_gaps",
    "material_usage_suggestion",
    "material_indexed_list",
    "inferred",
    "confidence",
]

_ALLOWED_PLATFORMS: Set[str] = {"抖音", "小红书", "B站", "快手", "微信视频号"}
_ALLOWED_MEDIA_TYPES: Set[str] = {
    "真人实拍",
    "动画",
    "插画",
    "数字人",
    "图文排版",
    "混合媒介",
}
_ALLOWED_CONFIDENCE: Set[str] = {"high", "medium", "low"}

_MATERIAL_VISION_SYSTEM = """你是短视频「素材分析」专用助手，只负责根据用户上传的图片做 material_analysis 判断。

硬性要求：
1. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释性文字。
2. 顶层结构必须为：{"material_analysis": { ... }}。
3. material_analysis 内必须且仅能包含以下键（不可缺省；数组没有内容时用 []，字符串没有内容时用 ""）：
material_status, material_count_estimate, material_types, available_elements,
usable_shot_directions, supported_presentation_modes, supported_narration_modes,
supported_content_directions, material_risks, material_gaps, material_usage_suggestion,
material_indexed_list, inferred, confidence
4. 有图片时 material_status 必须为 "has_material"；material_count_estimate 用字符串描述张数或估计。
5. 多张图时按用户上传顺序在 material_indexed_list 中编号为「素材[1]」「素材[2]」…
6. inferred 为布尔；confidence 取 high / medium / low。
7. 所有结论必须来自你实际看到的画面，不要编造画面中不存在的物体或文字。"""


def _apply_template(template: str, variables: Dict[str, str]) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _build_api1_system_prompt(variables: Dict[str, str]) -> str:
    raw = load_prompt("api1", "api1_input_analysis_result1.txt")
    return _apply_template(raw, variables)


def _mime_from_suffix(suffix: str) -> str:
    s = suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".png":
        return "image/png"
    if s == ".webp":
        return "image/webp"
    if s == ".gif":
        return "image/gif"
    return "image/png"


def _encode_images_as_data_urls(paths: List[str]) -> List[str]:
    urls: List[str] = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {p}")
        raw = path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        mime = _mime_from_suffix(path.suffix)
        urls.append(f"data:{mime};base64,{b64}")
    return urls


def _require_string(obj: Dict[str, Any], key: str, ctx: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise JSONParseError(f"{ctx}.{key} must be a string")
    return value.strip()


def _require_nonempty_str(obj: Dict[str, Any], key: str, ctx: str) -> str:
    value = _require_string(obj, key, ctx)
    if not value:
        raise JSONParseError(f"{ctx}.{key} must be a non-empty string")
    return value


def _normalize_confidence_level(value: Any, *, default: str = "high") -> str:
    raw = ""
    if value is None:
        raw = ""
    elif isinstance(value, str):
        raw = value.strip().lower()
    else:
        raw = str(value).strip().lower()

    if raw in {"high", "较高", "高", "很高", "high confidence"}:
        return "high"
    if raw in {"medium", "中", "中等", "一般", "medium confidence"}:
        return "medium"
    if raw in {"low", "较低", "低", "很低", "low confidence"}:
        return "low"

    return default


def _normalize_material_count_estimate(value: Any, *, has_uploaded_images: bool) -> str:
    if value is None:
        cleaned = ""
    elif isinstance(value, str):
        cleaned = value.strip()
    elif isinstance(value, (int, float, bool)):
        cleaned = str(value).strip()
    else:
        cleaned = ""

    if cleaned:
        return cleaned

    return "1" if has_uploaded_images else "0"


def _require_string_list(
    obj: Dict[str, Any],
    key: str,
    ctx: str,
    *,
    min_len: int = 0,
) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list):
        raise JSONParseError(f"{ctx}.{key} must be a list")
    if len(value) < min_len:
        raise JSONParseError(f"{ctx}.{key} must contain at least {min_len} items")
    cleaned: List[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise JSONParseError(f"{ctx}.{key}[{idx}] must be a non-empty string")
        cleaned.append(item.strip())
    return cleaned


def _validate_source_trace(source_trace: Any, *, has_uploaded_images: bool) -> None:
    if not isinstance(source_trace, dict):
        raise JSONParseError("result1.source_trace must be an object")
    validate_json(source_trace, ["from_user_input", "has_uploaded_images", "inferred_fields"])
    if source_trace.get("from_user_input") is not True:
        raise JSONParseError("result1.source_trace.from_user_input must be true")
    if source_trace.get("has_uploaded_images") is not has_uploaded_images:
        raise JSONParseError(
            f"result1.source_trace.has_uploaded_images must be {has_uploaded_images}"
        )
    _require_string_list(source_trace, "inferred_fields", "result1.source_trace")


def _validate_input_digest(input_digest: Any) -> None:
    if not isinstance(input_digest, dict):
        raise JSONParseError("result1.input_digest must be an object")
    validate_json(
        input_digest,
        [
            "topic",
            "target_platform",
            "duration_preference",
            "target_audience",
            "presentation_mode",
            "narration_mode",
            "user_goals",
            "extra_constraints",
        ],
    )
    _require_nonempty_str(input_digest, "topic", "result1.input_digest")
    platform = _require_nonempty_str(input_digest, "target_platform", "result1.input_digest")
    if platform not in _ALLOWED_PLATFORMS:
        raise JSONParseError(
            f"result1.input_digest.target_platform must be one of {sorted(_ALLOWED_PLATFORMS)}"
        )
    for key in [
        "duration_preference",
        "target_audience",
        "presentation_mode",
        "narration_mode",
    ]:
        _require_nonempty_str(input_digest, key, "result1.input_digest")
    _require_string_list(input_digest, "user_goals", "result1.input_digest", min_len=1)
    _require_string_list(input_digest, "extra_constraints", "result1.input_digest")


def _validate_core_decision(core: Any) -> None:
    if not isinstance(core, dict):
        raise JSONParseError("result1.core_decision must be an object")
    validate_json(
        core,
        [
            "core_expression",
            "primary_goal",
            "secondary_goal",
            "content_angle_summary",
            "primary_value_focus",
            "recommended_opening_direction",
        ],
    )
    for key in [
        "core_expression",
        "primary_goal",
        "secondary_goal",
        "content_angle_summary",
        "primary_value_focus",
        "recommended_opening_direction",
    ]:
        _require_nonempty_str(core, key, "result1.core_decision")


def _validate_theme_analysis(theme_analysis: Any) -> None:
    if not isinstance(theme_analysis, dict):
        raise JSONParseError("result1.theme_analysis must be an object")
    validate_json(
        theme_analysis,
        [
            "theme_core",
            "primary_scene",
            "primary_action",
            "key_elements",
            "possible_persona_traits",
            "theme_signal_summary",
        ],
    )
    for key in [
        "theme_core",
        "primary_scene",
        "primary_action",
        "theme_signal_summary",
    ]:
        _require_nonempty_str(theme_analysis, key, "result1.theme_analysis")

    _require_string_list(
        theme_analysis, "key_elements", "result1.theme_analysis", min_len=3
    )
    _require_string_list(
        theme_analysis,
        "possible_persona_traits",
        "result1.theme_analysis",
        min_len=2,
    )


def _validate_goal_semantics(goal_semantics: Any) -> None:
    if not isinstance(goal_semantics, dict):
        raise JSONParseError("result1.goal_semantics must be an object")
    validate_json(
        goal_semantics,
        [
            "primary_goal_keywords",
            "secondary_goal_keywords",
            "goal_expression_focus",
            "goal_to_theme_link",
        ],
    )
    _require_string_list(
        goal_semantics,
        "primary_goal_keywords",
        "result1.goal_semantics",
        min_len=3,
    )
    _require_string_list(
        goal_semantics,
        "secondary_goal_keywords",
        "result1.goal_semantics",
        min_len=1,
    )
    _require_nonempty_str(
        goal_semantics, "goal_expression_focus", "result1.goal_semantics"
    )
    _require_nonempty_str(
        goal_semantics, "goal_to_theme_link", "result1.goal_semantics"
    )


def _validate_condition_semantics(condition_semantics: Any) -> None:
    if not isinstance(condition_semantics, dict):
        raise JSONParseError("result1.condition_semantics must be an object")
    validate_json(
        condition_semantics,
        [
            "platform_semantics",
            "duration_semantics",
            "audience_semantics",
            "presentation_semantics",
            "narration_semantics",
            "production_semantics",
            "condition_to_execution_link",
        ],
    )
    for key in [
        "platform_semantics",
        "duration_semantics",
        "audience_semantics",
        "presentation_semantics",
        "narration_semantics",
        "production_semantics",
        "condition_to_execution_link",
    ]:
        _require_nonempty_str(condition_semantics, key, "result1.condition_semantics")


def _validate_platform_adaptation(platform_adaptation: Any) -> None:
    if not isinstance(platform_adaptation, dict):
        raise JSONParseError("result1.platform_adaptation must be an object")
    validate_json(
        platform_adaptation,
        [
            "target_platform",
            "platform_content_style",
            "recommended_opening_style",
            "recommended_rhythm_style",
            "recommended_expression_style",
            "platform_fit_reason",
            "platform_risk_note",
        ],
    )
    platform = _require_nonempty_str(
        platform_adaptation, "target_platform", "result1.platform_adaptation"
    )
    if platform not in _ALLOWED_PLATFORMS:
        raise JSONParseError(
            f"result1.platform_adaptation.target_platform must be one of {sorted(_ALLOWED_PLATFORMS)}"
        )
    for key in [
        "platform_content_style",
        "recommended_opening_style",
        "recommended_rhythm_style",
        "recommended_expression_style",
        "platform_fit_reason",
        "platform_risk_note",
    ]:
        _require_nonempty_str(platform_adaptation, key, "result1.platform_adaptation")


def _validate_audience_focus(audience_focus: Any) -> None:
    if not isinstance(audience_focus, dict):
        raise JSONParseError("result1.audience_focus must be an object")
    validate_json(
        audience_focus,
        [
            "primary_audience",
            "audience_mindset",
            "audience_value_expectation",
            "communication_note",
        ],
    )
    for key in [
        "primary_audience",
        "audience_mindset",
        "audience_value_expectation",
        "communication_note",
    ]:
        _require_nonempty_str(audience_focus, key, "result1.audience_focus")


def _validate_presentation_decision(presentation: Any) -> None:
    if not isinstance(presentation, dict):
        raise JSONParseError("result1.presentation_decision must be an object")
    validate_json(
        presentation,
        ["primary_presentation_mode", "presentation_reason", "execution_note"],
    )
    for key in ["primary_presentation_mode", "presentation_reason", "execution_note"]:
        _require_nonempty_str(presentation, key, "result1.presentation_decision")


def _validate_narration_decision(narration: Any) -> None:
    if not isinstance(narration, dict):
        raise JSONParseError("result1.narration_decision must be an object")
    validate_json(
        narration,
        ["primary_narration_mode", "narration_reason", "tone_direction"],
    )
    for key in ["primary_narration_mode", "narration_reason", "tone_direction"]:
        _require_nonempty_str(narration, key, "result1.narration_decision")


def _validate_material_analysis_block(block: Any, *, has_uploaded_images: bool) -> None:
    if not isinstance(block, dict):
        raise JSONParseError("result1.material_analysis must be an object")
    validate_json(block, _MATERIAL_ANALYSIS_KEYS)

    status = _require_nonempty_str(block, "material_status", "result1.material_analysis")

    material_count_estimate = _normalize_material_count_estimate(
        block.get("material_count_estimate"),
        has_uploaded_images=has_uploaded_images,
    )
    block["material_count_estimate"] = material_count_estimate

    for key in [
        "material_types",
        "available_elements",
        "usable_shot_directions",
        "supported_presentation_modes",
        "supported_narration_modes",
        "supported_content_directions",
        "material_risks",
        "material_gaps",
        "material_indexed_list",
    ]:
        _require_string_list(block, key, "result1.material_analysis")

    material_usage_suggestion = block.get("material_usage_suggestion")
    if not isinstance(material_usage_suggestion, str) or not material_usage_suggestion.strip():
        if has_uploaded_images:
            block["material_usage_suggestion"] = "优先使用现有素材组织内容，再补足必要镜头。"
        else:
            block["material_usage_suggestion"] = "当前无现成素材，建议优先采用低素材依赖的表达方式。"
    _require_nonempty_str(block, "material_usage_suggestion", "result1.material_analysis")

    inferred = block.get("inferred")
    if not isinstance(inferred, bool):
        raise JSONParseError("result1.material_analysis.inferred must be a boolean")

    confidence = _normalize_confidence_level(
        block.get("confidence"),
        default="high",
    )
    block["confidence"] = confidence
    if confidence not in _ALLOWED_CONFIDENCE:
        raise JSONParseError(
            "result1.material_analysis.confidence must be one of high / medium / low"
        )

    if has_uploaded_images and status != "has_material":
        raise JSONParseError(
            "result1.material_analysis.material_status must be 'has_material' when images are uploaded"
        )


def _validate_visual_style(visual_style: Any) -> None:
    if not isinstance(visual_style, dict):
        raise JSONParseError("result1.media_style_inference.visual_style must be an object")
    validate_json(visual_style, ["primary_style", "secondary_style", "style_summary"])
    _require_nonempty_str(visual_style, "primary_style", "result1.media_style_inference.visual_style")
    _require_string(visual_style, "secondary_style", "result1.media_style_inference.visual_style")
    _require_nonempty_str(visual_style, "style_summary", "result1.media_style_inference.visual_style")


def _validate_media_style_inference(media_style_inference: Any) -> None:
    if not isinstance(media_style_inference, dict):
        raise JSONParseError("result1.media_style_inference must be an object")
    validate_json(
        media_style_inference,
        [
            "recommended_media_type",
            "confidence_level",
            "reasoning_summary",
            "visual_style",
            "prompt_dimension_profile",
        ],
    )

    media_type = _require_nonempty_str(
        media_style_inference,
        "recommended_media_type",
        "result1.media_style_inference",
    )
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise JSONParseError(
            f"result1.media_style_inference.recommended_media_type must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        )

    confidence_level = _normalize_confidence_level(
        media_style_inference.get("confidence_level"),
        default="high",
    )
    media_style_inference["confidence_level"] = confidence_level
    if confidence_level not in _ALLOWED_CONFIDENCE:
        raise JSONParseError(
            "result1.media_style_inference.confidence_level must be one of high / medium / low"
        )

    reasoning_summary = media_style_inference.get("reasoning_summary")
    if not isinstance(reasoning_summary, str) or not reasoning_summary.strip():
        media_style_inference["reasoning_summary"] = "当前媒介类型更适合承接该主题的表达重点。"
    _require_nonempty_str(
        media_style_inference,
        "reasoning_summary",
        "result1.media_style_inference",
    )

    _validate_visual_style(media_style_inference.get("visual_style"))

    prompt_dimension_profile = _require_string_list(
        media_style_inference,
        "prompt_dimension_profile",
        "result1.media_style_inference",
        min_len=3,
    )
    media_style_inference["prompt_dimension_profile"] = prompt_dimension_profile[:8]


def _validate_execution_guidance(execution_guidance: Any) -> None:
    if not isinstance(execution_guidance, dict):
        raise JSONParseError("result1.execution_guidance must be an object")
    validate_json(
        execution_guidance,
        [
            "recommended_structure_direction",
            "material_usage_direction",
            "execution_complexity",
            "must_keep_elements",
            "avoid_elements",
            "risk_notes",
        ],
    )
    for key in [
        "recommended_structure_direction",
        "material_usage_direction",
        "execution_complexity",
    ]:
        _require_nonempty_str(execution_guidance, key, "result1.execution_guidance")
    _require_string_list(execution_guidance, "must_keep_elements", "result1.execution_guidance")
    _require_string_list(execution_guidance, "avoid_elements", "result1.execution_guidance")
    _require_string_list(execution_guidance, "risk_notes", "result1.execution_guidance")


def _validate_risk_and_confirmation(risk_and_confirmation: Any) -> None:
    if not isinstance(risk_and_confirmation, dict):
        raise JSONParseError("result1.risk_and_confirmation must be an object")
    validate_json(
        risk_and_confirmation,
        ["main_risk", "needs_user_confirmation", "weak_assumptions"],
    )
    _require_nonempty_str(risk_and_confirmation, "main_risk", "result1.risk_and_confirmation")
    _require_string_list(
        risk_and_confirmation, "needs_user_confirmation", "result1.risk_and_confirmation"
    )
    _require_string_list(
        risk_and_confirmation, "weak_assumptions", "result1.risk_and_confirmation"
    )


def _validate_editable_fields(editable_fields: Any) -> None:
    if not isinstance(editable_fields, list) or len(editable_fields) < 6:
        raise JSONParseError("result1.editable_fields must be a list with at least 6 items")

    seen_paths: Set[str] = set()
    for idx, item in enumerate(editable_fields):
        ctx = f"result1.editable_fields[{idx}]"
        if not isinstance(item, dict):
            raise JSONParseError(f"{ctx} must be an object")
        validate_json(item, ["field_path", "field_label", "current_value"])
        path = _require_nonempty_str(item, "field_path", ctx)
        if path in seen_paths:
            raise JSONParseError("result1.editable_fields.field_path must not repeat")
        seen_paths.add(path)
        _require_nonempty_str(item, "field_label", ctx)
        _require_string(item, "current_value", ctx)


def _first_nonempty(*values: object) -> str:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _clean_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text and text not in cleaned:
                cleaned.append(text)
    return cleaned


_GOAL_KEYWORD_MAP: Dict[str, List[str]] = {
    "建立人设": ["个人标签", "记忆点", "稳定风格", "人设识别"],
    "引发共鸣": ["情绪代入", "拉近距离", "共鸣感", "互动讨论"],
    "教程教学": ["步骤清楚", "信息密度", "可复用", "解释到位"],
    "分享观点": ["表达判断", "观点输出", "讨论空间", "态度鲜明"],
    "记录生活": ["真实日常", "生活场景", "细节氛围", "轻记录感"],
    "提升完播": ["开头抓人", "推进清楚", "节奏稳定", "持续观看"],
    "提升互动": ["讨论引导", "情绪触发", "可回应点", "评论欲望"],
}


def _keywords_from_goal_text(text: str, *, fallback: str) -> List[str]:
    source = (text or "").strip()
    results: List[str] = []

    for goal_label, keywords in _GOAL_KEYWORD_MAP.items():
        if goal_label in source:
            for kw in keywords:
                if kw not in results:
                    results.append(kw)

    if not results:
        default_map = _GOAL_KEYWORD_MAP.get(fallback, ["贴合主题", "表达清楚", "便于展开"])
        results.extend(default_map)

    return results[:5]


def _normalize_theme_analysis(
    r1: Dict[str, Any],
    *,
    raw_topic: str,
) -> Dict[str, Any]:
    theme = r1.get("theme_analysis")
    if not isinstance(theme, dict):
        theme = {}

    input_digest = r1.get("input_digest", {}) if isinstance(r1.get("input_digest"), dict) else {}
    core_decision = r1.get("core_decision", {}) if isinstance(r1.get("core_decision"), dict) else {}

    topic = _first_nonempty(raw_topic, input_digest.get("topic"), core_decision.get("core_expression"), "当前主题")

    theme["theme_core"] = _first_nonempty(
        theme.get("theme_core"),
        core_decision.get("core_expression"),
        f"这是一条围绕“{topic}”展开的短视频内容。",
    )
    theme["primary_scene"] = _first_nonempty(
        theme.get("primary_scene"),
        topic,
        "主题相关场景",
    )
    theme["primary_action"] = _first_nonempty(
        theme.get("primary_action"),
        "围绕主题进行记录、表达或分享",
    )

    key_elements = _clean_string_list(theme.get("key_elements"))
    if len(key_elements) < 3:
        key_elements = [topic, "场景细节", "内容重点"]
    theme["key_elements"] = key_elements[:6]

    persona_traits = _clean_string_list(theme.get("possible_persona_traits"))
    if len(persona_traits) < 2:
        persona_traits = ["真实表达", "持续输出"]
    theme["possible_persona_traits"] = persona_traits[:5]

    theme["theme_signal_summary"] = _first_nonempty(
        theme.get("theme_signal_summary"),
        f"这条主题最值得放大的是“{topic}”背后的场景感与表达意图。",
    )
    return theme


def _normalize_goal_semantics(r1: Dict[str, Any]) -> Dict[str, Any]:
    goal_sem = r1.get("goal_semantics")
    if not isinstance(goal_sem, dict):
        goal_sem = {}

    core_decision = r1.get("core_decision", {}) if isinstance(r1.get("core_decision"), dict) else {}
    primary_goal = _first_nonempty(core_decision.get("primary_goal"), "讲清主题")
    secondary_goal = _first_nonempty(core_decision.get("secondary_goal"), "无明确次目标")

    primary_keywords = _clean_string_list(goal_sem.get("primary_goal_keywords"))
    if len(primary_keywords) < 3:
        primary_keywords = _keywords_from_goal_text(primary_goal, fallback="分享观点")

    secondary_keywords = _clean_string_list(goal_sem.get("secondary_goal_keywords"))
    if len(secondary_keywords) < 1:
        secondary_keywords = _keywords_from_goal_text(secondary_goal, fallback="记录生活")[:3]

    goal_sem["primary_goal_keywords"] = primary_keywords[:5]
    goal_sem["secondary_goal_keywords"] = secondary_keywords[:4]
    goal_sem["goal_expression_focus"] = _first_nonempty(
        goal_sem.get("goal_expression_focus"),
        f"表达重点应围绕“{primary_goal}”展开，并兼顾“{secondary_goal}”。",
    )
    goal_sem["goal_to_theme_link"] = _first_nonempty(
        goal_sem.get("goal_to_theme_link"),
        "这些目标需要作用到当前主题的切入方式、表达重心与人物/内容呈现上。",
    )
    return goal_sem


def _normalize_condition_semantics(r1: Dict[str, Any]) -> Dict[str, Any]:
    cond = r1.get("condition_semantics")
    if not isinstance(cond, dict):
        cond = {}

    input_digest = r1.get("input_digest", {}) if isinstance(r1.get("input_digest"), dict) else {}
    platform_adaptation = r1.get("platform_adaptation", {}) if isinstance(r1.get("platform_adaptation"), dict) else {}
    audience_focus = r1.get("audience_focus", {}) if isinstance(r1.get("audience_focus"), dict) else {}
    presentation_decision = r1.get("presentation_decision", {}) if isinstance(r1.get("presentation_decision"), dict) else {}
    narration_decision = r1.get("narration_decision", {}) if isinstance(r1.get("narration_decision"), dict) else {}
    execution_guidance = r1.get("execution_guidance", {}) if isinstance(r1.get("execution_guidance"), dict) else {}

    cond["platform_semantics"] = _first_nonempty(
        cond.get("platform_semantics"),
        platform_adaptation.get("platform_content_style"),
        "平台会影响内容的表达风格与可信感建立方式。",
    )
    cond["duration_semantics"] = _first_nonempty(
        cond.get("duration_semantics"),
        f"当前时长偏向“{_first_nonempty(input_digest.get('duration_preference'), '短视频表达')}”，需要控制信息密度与推进节奏。",
    )
    cond["audience_semantics"] = _first_nonempty(
        cond.get("audience_semantics"),
        audience_focus.get("communication_note"),
        "表达要匹配当前受众的理解方式与期待。",
    )
    cond["presentation_semantics"] = _first_nonempty(
        cond.get("presentation_semantics"),
        presentation_decision.get("presentation_reason"),
        "出镜方式会直接影响画面组织与人物呈现。",
    )
    cond["narration_semantics"] = _first_nonempty(
        cond.get("narration_semantics"),
        narration_decision.get("narration_reason"),
        "表达方式会直接影响台词、字幕与节奏组织。",
    )
    cond["production_semantics"] = _first_nonempty(
        cond.get("production_semantics"),
        execution_guidance.get("execution_complexity"),
        "制作偏好会影响执行复杂度与内容包装力度。",
    )
    cond["condition_to_execution_link"] = _first_nonempty(
        cond.get("condition_to_execution_link"),
        "这些条件会共同作用到后续方案分叉、脚本组织和 AI 指令展开方式。",
    )
    return cond


def _normalize_api1_output(
    data: Dict[str, Any],
    *,
    raw_topic: str,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}

    if not isinstance(data.get("result1"), dict):
        data["result1"] = {}

    r1 = data["result1"]
    r1["theme_analysis"] = _normalize_theme_analysis(r1, raw_topic=raw_topic)
    r1["goal_semantics"] = _normalize_goal_semantics(r1)
    r1["condition_semantics"] = _normalize_condition_semantics(r1)
    return data


def _validate_api1_output(data: Dict[str, Any], *, has_uploaded_images: bool) -> None:
    validate_json(data, ["result1"])
    validate_json(data["result1"], _RESULT1_INNER_KEYS)

    r1 = data["result1"]
    schema_version = _require_nonempty_str(r1, "schema_version", "result1")
    if schema_version != "v2.0":
        raise JSONParseError("result1.schema_version must be 'v2.0'")

    _validate_source_trace(r1.get("source_trace"), has_uploaded_images=has_uploaded_images)
    _validate_input_digest(r1.get("input_digest"))
    _validate_core_decision(r1.get("core_decision"))
    _validate_theme_analysis(r1.get("theme_analysis"))
    _validate_goal_semantics(r1.get("goal_semantics"))
    _validate_condition_semantics(r1.get("condition_semantics"))
    _validate_platform_adaptation(r1.get("platform_adaptation"))
    _validate_audience_focus(r1.get("audience_focus"))
    _validate_presentation_decision(r1.get("presentation_decision"))
    _validate_narration_decision(r1.get("narration_decision"))
    _validate_material_analysis_block(
        r1.get("material_analysis"), has_uploaded_images=has_uploaded_images
    )
    _validate_media_style_inference(r1.get("media_style_inference"))
    _validate_execution_guidance(r1.get("execution_guidance"))
    _validate_risk_and_confirmation(r1.get("risk_and_confirmation"))
    _validate_editable_fields(r1.get("editable_fields"))

    input_platform = r1["input_digest"]["target_platform"]
    adapted_platform = r1["platform_adaptation"]["target_platform"]
    if input_platform != adapted_platform:
        raise JSONParseError(
            "result1.input_digest.target_platform must equal result1.platform_adaptation.target_platform"
        )


async def _vision_material_analysis_only(
    client: AIClient,
    paths: List[str],
) -> Dict[str, Any]:
    """多模态：仅产出 material_analysis，结构与 result1 中该字段完全一致。"""
    names = "、".join(Path(p).expanduser().name for p in paths)
    user_text = (
        f"共 {len(paths)} 张素材图（文件名供参考：{names}）。"
        "请逐张查看图片内容，完成 material_analysis。"
        "只输出 JSON。"
    )
    data_urls = _encode_images_as_data_urls(paths)
    raw = await client.request_vision(
        _MATERIAL_VISION_SYSTEM,
        user_text,
        data_urls,
        response_format=JSON_OBJECT_FORMAT,
        model=_VISION_MODEL,
    )
    parsed = parse_json_from_llm(raw)
    validate_json(parsed, ["material_analysis"])
    material_analysis = parsed["material_analysis"]
    _validate_material_analysis_block(material_analysis, has_uploaded_images=True)
    return material_analysis


async def run_api1(
    *,
    topic: str,
    goals: str = "",
    platforms: str = "",
    duration: str = "",
    audiences: str = "",
    presentation_mode: str = "",
    narration_mode: str = "",
    materials: str = "",
    extra_notes: str = "",
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    client = AIClient(api_stage="api1")
    paths = [x.strip() for x in (image_paths or []) if x and str(x).strip()]

    mat = materials.strip() if materials else ""
    if paths:
        names = "、".join(Path(p).expanduser().name for p in paths)
        note = (
            f"（用户已通过 CLI 上传 {len(paths)} 张素材图：{names}。"
            "画面内容已由 gpt-4o-mini 多模态单独分析并写入服务端 material_analysis；"
            "你生成 result1 时其它维度请结合「确有可用素材」这一事实，"
            "material_analysis 字段服务端会用视觉结果覆盖，你可按无图逻辑简要占位或合理推断，不必复述画面细节。）"
        )
        mat = f"{mat}\n{note}".strip() if mat else note

    variables = {
        "topic": topic,
        "goals": goals,
        "platforms": platforms,
        "duration": duration,
        "audiences": audiences,
        "presentation_mode": presentation_mode,
        "narration_mode": narration_mode,
        "materials": mat,
        "extra_notes": extra_notes,
    }
    system = _build_api1_system_prompt(variables)
    raw_topic = topic.strip() or "未命名主题"

    if paths:
        material_from_vision = await _vision_material_analysis_only(client, paths)
        raw = await client.request_chat(
            system, USER_FOOTER, response_format=JSON_OBJECT_FORMAT
        )
        data = parse_json_from_llm(raw)
        if isinstance(data.get("result1"), dict):
            result1_obj = data["result1"]
            if not isinstance(result1_obj.get("input_digest"), dict):
                result1_obj["input_digest"] = {}
            result1_obj["input_digest"]["topic"] = raw_topic
            result1_obj["material_analysis"] = material_from_vision
            if isinstance(result1_obj.get("source_trace"), dict):
                result1_obj["source_trace"]["has_uploaded_images"] = True
    else:
        raw = await client.request_chat(
            system, USER_FOOTER, response_format=JSON_OBJECT_FORMAT
        )
        data = parse_json_from_llm(raw)
        if isinstance(data.get("result1"), dict):
            result1_obj = data["result1"]
            if not isinstance(result1_obj.get("input_digest"), dict):
                result1_obj["input_digest"] = {}
            result1_obj["input_digest"]["topic"] = raw_topic

    data = _normalize_api1_output(
        data,
        raw_topic=raw_topic,
    )
    _validate_api1_output(data, has_uploaded_images=bool(paths))
    return data