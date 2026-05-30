"""API3: selected option → result3.

与 API1 相同的 OpenAI 调用链：传入的 ``AIClient(api_stage="api3")`` 使用
``AsyncOpenAI.chat.completions``、``response_format={"type": "json_object"}``；
凭据读取 ``API_KEY`` / ``MODEL`` / ``API_BASE_URL``（缺省时回退 ``API1_*``）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from api.ai_client import AIClient
from utils.json_parser import JSONParseError, parse_json_from_llm, validate_json
from utils.prompt_loader import combine_prompts, load_prompt

JSON_OBJECT_FORMAT: Dict[str, str] = {"type": "json_object"}

USER_FOOTER = (
    "以上为完整任务说明与占位符已替换后的输入。"
    "请严格只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释性文字。"
)

_RESULT3_INNER_KEYS: List[str] = [
    "schema_version",
    "source_trace",
    "generation_route",
    "selected_option_id",
    "selected_option_name",
    "based_on_edited_result1",
    "based_on_selected_option",
    "base_storyboard_script",
    "route_display_data",
]

_STORYBOARD_COLUMNS: List[str] = [
    "镜号",
    "时间轴",
    "画面描述",
    "机位",
    "摄法",
    "景别",
    "台词",
    "音效",
]

_ALLOWED_ROUTES: Set[str] = {"ai_instruction", "pro_script"}
_ALLOWED_PLATFORMS: Set[str] = {"抖音", "小红书", "B站", "快手", "微信视频号"}
_ALLOWED_MEDIA_TYPES: Set[str] = {
    "真人实拍",
    "动画",
    "插画",
    "数字人",
    "图文排版",
    "混合媒介",
}

_TIME_RANGE_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _apply_template(template: str, variables: Dict[str, str]) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _unwrap_result1_blob(blob: Dict[str, Any]) -> Dict[str, Any]:
    if "result1" in blob and isinstance(blob["result1"], dict):
        return blob["result1"]
    return blob


def _unwrap_result2_blob(blob: Dict[str, Any]) -> Dict[str, Any]:
    if "result2" in blob and isinstance(blob["result2"], dict):
        return blob["result2"]
    return blob


def _build_api3_system_prompt(generation_route: str, variables: Dict[str, str]) -> str:
    if generation_route == "ai_instruction":
        route_file = "api3_route_ai_instruction_expand.txt"
    elif generation_route == "pro_script":
        route_file = "api3_route_pro_script_expand.txt"
    else:
        raise ValueError(
            "generation_route must be 'ai_instruction' or 'pro_script', "
            f"got {generation_route!r}"
        )
    base = load_prompt("api3", "api3_shared_base.txt")
    route = load_prompt("api3", route_file)
    combined = combine_prompts([base, route])
    return _apply_template(combined, variables)


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


def _timecode_to_seconds(timecode: str) -> Tuple[int, int]:
    match = _TIME_RANGE_RE.match(timecode)
    if not match:
        raise JSONParseError(
            f"Invalid timecode format {timecode!r}; expected HH:MM-HH:MM"
        )
    sm, ss, em, es = map(int, match.groups())
    start = sm * 60 + ss
    end = em * 60 + es
    if end <= start:
        raise JSONParseError(f"Invalid time range {timecode!r}; end must be after start")
    return start, end


def _validate_visual_style(style: Any) -> None:
    if not isinstance(style, dict):
        raise JSONParseError("base_storyboard_script.visual_style must be an object")
    validate_json(style, ["primary_style", "secondary_style", "style_summary"])
    _require_nonempty_str(style, "primary_style", "base_storyboard_script.visual_style")
    _require_string(style, "secondary_style", "base_storyboard_script.visual_style")
    _require_nonempty_str(style, "style_summary", "base_storyboard_script.visual_style")


def _validate_prompt_dimension_profile(profile: Any) -> None:
    if not isinstance(profile, list) or len(profile) < 2:
        raise JSONParseError(
            "base_storyboard_script.prompt_dimension_profile must be a list with at least 2 items"
        )
    for idx, item in enumerate(profile):
        if not isinstance(item, str) or not item.strip():
            raise JSONParseError(
                f"base_storyboard_script.prompt_dimension_profile[{idx}] must be a non-empty string"
            )


def _validate_storyboard_shots(shots: Any) -> List[int]:
    if not isinstance(shots, list) or len(shots) < 3:
        raise JSONParseError("base_storyboard_script.shots must be a list with at least 3 items")

    prev_end: Optional[int] = None
    ordered_numbers: List[int] = []
    required = [
        "shot_no",
        "timecode",
        "visual_description",
        "camera_position",
        "shooting_method",
        "shot_size",
        "dialogue",
        "sound_effect",
    ]

    for idx, shot in enumerate(shots):
        ctx = f"base_storyboard_script.shots[{idx}]"
        if not isinstance(shot, dict):
            raise JSONParseError(f"{ctx} must be an object")
        validate_json(shot, required)

        shot_no = shot.get("shot_no")
        if not isinstance(shot_no, int) or shot_no <= 0:
            raise JSONParseError(f"{ctx}.shot_no must be a positive integer")
        ordered_numbers.append(shot_no)

        start, end = _timecode_to_seconds(str(shot.get("timecode", "")))
        if prev_end is not None and start != prev_end:
            raise JSONParseError(
                f"{ctx}.timecode must be continuous with previous shot; expected start={prev_end}, got {start}"
            )
        prev_end = end

        for key in required[2:]:
            _require_nonempty_str(shot, key, ctx)

    expected = list(range(1, len(ordered_numbers) + 1))
    if ordered_numbers != expected:
        raise JSONParseError(
            "shot_no must be continuous starting from 1 in base_storyboard_script.shots"
        )
    return ordered_numbers


def _validate_segment_groups(segment_groups: Any, ordered_shot_numbers: Sequence[int]) -> None:
    if not isinstance(segment_groups, list) or len(segment_groups) < 3:
        raise JSONParseError(
            "base_storyboard_script.segment_groups must be a list with at least 3 items"
        )

    expected_shot_set = set(ordered_shot_numbers)
    covered: List[int] = []
    seen_segment_ids: Set[str] = set()
    prev_last_shot: Optional[int] = None

    required = [
        "segment_id",
        "segment_name",
        "shot_range",
        "related_shot_numbers",
        "segment_goal",
        "segment_summary",
    ]

    for idx, segment in enumerate(segment_groups):
        ctx = f"base_storyboard_script.segment_groups[{idx}]"
        if not isinstance(segment, dict):
            raise JSONParseError(f"{ctx} must be an object")
        validate_json(segment, required)

        segment_id = _require_nonempty_str(segment, "segment_id", ctx)
        if segment_id in seen_segment_ids:
            raise JSONParseError(f"Duplicate segment_id found: {segment_id}")
        seen_segment_ids.add(segment_id)

        _require_nonempty_str(segment, "segment_name", ctx)
        shot_range = _require_nonempty_str(segment, "shot_range", ctx)
        _require_nonempty_str(segment, "segment_goal", ctx)
        _require_nonempty_str(segment, "segment_summary", ctx)

        related = segment.get("related_shot_numbers")
        if not isinstance(related, list) or not related:
            raise JSONParseError(f"{ctx}.related_shot_numbers must be a non-empty list")
        if any(not isinstance(n, int) or n <= 0 for n in related):
            raise JSONParseError(f"{ctx}.related_shot_numbers must contain positive integers")

        ordered_related = sorted(related)
        if related != ordered_related:
            raise JSONParseError(f"{ctx}.related_shot_numbers must be in ascending order")
        if ordered_related != list(range(ordered_related[0], ordered_related[-1] + 1)):
            raise JSONParseError(f"{ctx}.related_shot_numbers must be continuous")

        for shot_no in related:
            if shot_no not in expected_shot_set:
                raise JSONParseError(f"{ctx} references unknown shot_no {shot_no}")

        if prev_last_shot is not None and related[0] != prev_last_shot + 1:
            raise JSONParseError(
                f"{ctx}.related_shot_numbers must continue from previous segment"
            )
        prev_last_shot = related[-1]
        covered.extend(related)

        start_no = related[0]
        end_no = related[-1]
        expected_range = f"镜头{start_no}-{end_no}"
        allowed_ranges = {expected_range}

        if start_no == end_no:
            allowed_ranges.add(f"镜头{start_no}")

        if shot_range not in allowed_ranges:
            raise JSONParseError(
                f"{ctx}.shot_range must be one of {sorted(allowed_ranges)!r}, got {shot_range!r}"
            )

    if covered != list(ordered_shot_numbers):
        raise JSONParseError(
            "segment_groups.related_shot_numbers must partition all shots in order without gaps or overlaps"
        )


def _validate_base_storyboard_script(base: Any) -> None:
    if not isinstance(base, dict):
        raise JSONParseError("base_storyboard_script must be an object")
    validate_json(
        base,
        [
            "script_title",
            "one_line_concept",
            "target_platform",
            "total_duration",
            "recommended_media_type",
            "visual_style",
            "prompt_dimension_profile",
            "table_columns",
            "shots",
            "segment_groups",
        ],
    )

    _require_nonempty_str(base, "script_title", "base_storyboard_script")
    _require_nonempty_str(base, "one_line_concept", "base_storyboard_script")
    platform = _require_nonempty_str(base, "target_platform", "base_storyboard_script")
    if platform not in _ALLOWED_PLATFORMS:
        raise JSONParseError(
            f"base_storyboard_script.target_platform must be one of {sorted(_ALLOWED_PLATFORMS)}"
        )

    _require_nonempty_str(base, "total_duration", "base_storyboard_script")
    media_type = _require_nonempty_str(base, "recommended_media_type", "base_storyboard_script")
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise JSONParseError(
            f"base_storyboard_script.recommended_media_type must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        )

    _validate_visual_style(base.get("visual_style"))
    _validate_prompt_dimension_profile(base.get("prompt_dimension_profile"))

    columns = base.get("table_columns")
    if columns != _STORYBOARD_COLUMNS:
        raise JSONParseError(
            "base_storyboard_script.table_columns must exactly equal the fixed storyboard columns"
        )

    ordered_shot_numbers = _validate_storyboard_shots(base.get("shots"))
    _validate_segment_groups(base.get("segment_groups"), ordered_shot_numbers)


def _validate_pro_script_display(display: Any) -> None:
    if not isinstance(display, dict):
        raise JSONParseError("route_display_data must be an object")
    validate_json(
        display,
        [
            "display_type",
            "page_title",
            "page_subtitle",
            "primary_data_binding",
            "table_presentation",
            "editing_config",
            "export_config",
        ],
    )

    if display.get("display_type") != "storyboard_table":
        raise JSONParseError("pro_script route_display_data.display_type must be 'storyboard_table'")
    if _require_nonempty_str(display, "page_title", "route_display_data") != "专业脚本":
        raise JSONParseError("pro_script route_display_data.page_title must be '专业脚本'")
    _require_string(display, "page_subtitle", "route_display_data")

    binding = display.get("primary_data_binding")
    if not isinstance(binding, dict):
        raise JSONParseError("route_display_data.primary_data_binding must be an object")
    validate_json(binding, ["table_columns_path", "table_rows_path"])
    if binding.get("table_columns_path") != "base_storyboard_script.table_columns":
        raise JSONParseError(
            "primary_data_binding.table_columns_path must be 'base_storyboard_script.table_columns'"
        )
    if binding.get("table_rows_path") != "base_storyboard_script.shots":
        raise JSONParseError(
            "primary_data_binding.table_rows_path must be 'base_storyboard_script.shots'"
        )

    table_presentation = display.get("table_presentation")
    if not isinstance(table_presentation, dict):
        raise JSONParseError("route_display_data.table_presentation must be an object")
    validate_json(
        table_presentation,
        ["table_name", "default_view_mode", "show_full_table_directly", "column_order"],
    )
    _require_nonempty_str(table_presentation, "table_name", "route_display_data.table_presentation")
    if table_presentation.get("default_view_mode") != "table":
        raise JSONParseError("table_presentation.default_view_mode must be 'table'")
    if table_presentation.get("show_full_table_directly") is not True:
        raise JSONParseError("table_presentation.show_full_table_directly must be true")
    if table_presentation.get("column_order") != _STORYBOARD_COLUMNS:
        raise JSONParseError(
            "table_presentation.column_order must exactly equal the fixed storyboard columns"
        )

    editing_config = display.get("editing_config")
    if not isinstance(editing_config, dict):
        raise JSONParseError("route_display_data.editing_config must be an object")
    validate_json(editing_config, ["editable", "edit_entry", "editable_scope", "editable_fields"])
    if editing_config.get("editable") is not True:
        raise JSONParseError("editing_config.editable must be true")
    if editing_config.get("edit_entry") != "one_click_edit":
        raise JSONParseError("editing_config.edit_entry must be 'one_click_edit'")
    if editing_config.get("editable_scope") != "all_cells":
        raise JSONParseError("editing_config.editable_scope must be 'all_cells'")
    expected_editable_fields = [
        "shot_no",
        "timecode",
        "visual_description",
        "camera_position",
        "shooting_method",
        "shot_size",
        "dialogue",
        "sound_effect",
    ]
    if editing_config.get("editable_fields") != expected_editable_fields:
        raise JSONParseError("editing_config.editable_fields does not match the required storyboard field keys")

    export_config = display.get("export_config")
    if not isinstance(export_config, dict):
        raise JSONParseError("route_display_data.export_config must be an object")
    validate_json(export_config, ["export_enabled", "supported_formats", "default_file_name"])
    if export_config.get("export_enabled") is not True:
        raise JSONParseError("export_config.export_enabled must be true")
    if export_config.get("supported_formats") != ["jpg", "pdf", "excel"]:
        raise JSONParseError(
            "export_config.supported_formats must be exactly ['jpg', 'pdf', 'excel']"
        )
    _require_nonempty_str(export_config, "default_file_name", "route_display_data.export_config")


def _validate_ai_instruction_display(display: Any, base: Dict[str, Any]) -> None:
    if not isinstance(display, dict):
        raise JSONParseError("route_display_data must be an object")
    validate_json(
        display,
        [
            "display_type",
            "page_title",
            "page_subtitle",
            "primary_source_binding",
            "instruction_segments",
            "full_instruction_package",
        ],
    )
    if display.get("display_type") != "segmented_ai_instructions":
        raise JSONParseError(
            "ai_instruction route_display_data.display_type must be 'segmented_ai_instructions'"
        )
    if _require_nonempty_str(display, "page_title", "route_display_data") != "AI指令":
        raise JSONParseError("ai_instruction route_display_data.page_title must be 'AI指令'")
    _require_string(display, "page_subtitle", "route_display_data")

    binding = display.get("primary_source_binding")
    if not isinstance(binding, dict):
        raise JSONParseError("route_display_data.primary_source_binding must be an object")
    validate_json(binding, ["segments_path", "shots_path"])
    if binding.get("segments_path") != "base_storyboard_script.segment_groups":
        raise JSONParseError(
            "primary_source_binding.segments_path must be 'base_storyboard_script.segment_groups'"
        )
    if binding.get("shots_path") != "base_storyboard_script.shots":
        raise JSONParseError(
            "primary_source_binding.shots_path must be 'base_storyboard_script.shots'"
        )

    expected_segments = base.get("segment_groups", [])
    cards = display.get("instruction_segments")
    if not isinstance(cards, list) or len(cards) != len(expected_segments):
        raise JSONParseError(
            "route_display_data.instruction_segments must match base_storyboard_script.segment_groups in length"
        )

    for idx, (segment, card) in enumerate(zip(expected_segments, cards)):
        ctx = f"route_display_data.instruction_segments[{idx}]"
        if not isinstance(segment, dict):
            raise JSONParseError(f"base_storyboard_script.segment_groups[{idx}] must be an object")
        if not isinstance(card, dict):
            raise JSONParseError(f"{ctx} must be an object")
        validate_json(
            card,
            [
                "segment_id",
                "segment_name",
                "shot_range",
                "related_shot_numbers",
                "segment_goal",
                "instruction_text",
                "spoken_lines",
                "subtitle_focus",
                "copy_enabled",
            ],
        )
        if card.get("segment_id") != segment.get("segment_id"):
            raise JSONParseError(f"{ctx}.segment_id must match base_storyboard_script.segment_groups[{idx}]")
        if card.get("segment_name") != segment.get("segment_name"):
            raise JSONParseError(f"{ctx}.segment_name must match base_storyboard_script.segment_groups[{idx}]")
        if card.get("shot_range") != segment.get("shot_range"):
            raise JSONParseError(f"{ctx}.shot_range must match base_storyboard_script.segment_groups[{idx}]")
        if card.get("related_shot_numbers") != segment.get("related_shot_numbers"):
            raise JSONParseError(
                f"{ctx}.related_shot_numbers must match base_storyboard_script.segment_groups[{idx}]"
            )
        if card.get("segment_goal") != segment.get("segment_goal"):
            raise JSONParseError(f"{ctx}.segment_goal must match base_storyboard_script.segment_groups[{idx}]")

        text = _require_nonempty_str(card, "instruction_text", ctx)
        if len(text) < 60:
            raise JSONParseError(f"{ctx}.instruction_text is too short; need at least 60 characters")

        spoken_lines = card.get("spoken_lines")
        if not isinstance(spoken_lines, list) or len(spoken_lines) < 1:
            raise JSONParseError(f"{ctx}.spoken_lines must be a non-empty list")
        for line_idx, line in enumerate(spoken_lines):
            if not isinstance(line, str) or not line.strip():
                raise JSONParseError(f"{ctx}.spoken_lines[{line_idx}] must be a non-empty string")

        subtitle_focus = card.get("subtitle_focus")
        if not isinstance(subtitle_focus, list) or len(subtitle_focus) < 1:
            raise JSONParseError(f"{ctx}.subtitle_focus must be a non-empty list")
        if len(subtitle_focus) > 3:
            raise JSONParseError(f"{ctx}.subtitle_focus must contain at most 3 items")
        for sub_idx, item in enumerate(subtitle_focus):
            if not isinstance(item, str) or not item.strip():
                raise JSONParseError(f"{ctx}.subtitle_focus[{sub_idx}] must be a non-empty string")

        if card.get("copy_enabled") is not True:
            raise JSONParseError(f"{ctx}.copy_enabled must be true")

    full_pkg = display.get("full_instruction_package")
    if not isinstance(full_pkg, dict):
        raise JSONParseError("route_display_data.full_instruction_package must be an object")
    validate_json(full_pkg, ["title", "full_instruction_text", "copy_enabled"])
    _require_nonempty_str(full_pkg, "title", "route_display_data.full_instruction_package")
    full_text = _require_nonempty_str(
        full_pkg,
        "full_instruction_text",
        "route_display_data.full_instruction_package",
    )
    if len(full_text) < 180:
        raise JSONParseError(
            "route_display_data.full_instruction_package.full_instruction_text is too short; need at least 180 characters"
        )
    if full_pkg.get("copy_enabled") is not True:
        raise JSONParseError("route_display_data.full_instruction_package.copy_enabled must be true")


def _first_nonempty(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_user_narration_mode(result1: Dict[str, Any]) -> str:
    r1 = result1.get("result1", {}) if isinstance(result1, dict) else {}
    input_digest = r1.get("input_digest", {}) if isinstance(r1.get("input_digest"), dict) else {}

    return _first_nonempty(
        input_digest.get("narration_mode"),
        input_digest.get("primary_narration_mode"),
    )


def _is_explicit_subtitle_music(result1: Dict[str, Any]) -> bool:
    return _resolve_user_narration_mode(result1) == "纯字幕+音乐"


def _fill_storyboard_required_fields(
    data: dict,
    *,
    result1: dict,
    result2: dict,
    selected_option_id: str,
) -> None:
    if not isinstance(data.get("result3"), dict):
        return

    result3_obj = data["result3"]

    if not isinstance(result3_obj.get("base_storyboard_script"), dict):
        result3_obj["base_storyboard_script"] = {}

    base = result3_obj["base_storyboard_script"]

    r1 = result1.get("result1", {}) if isinstance(result1, dict) else {}
    r2 = result2.get("result2", {}) if isinstance(result2, dict) else {}

    input_digest = r1.get("input_digest", {}) if isinstance(r1.get("input_digest"), dict) else {}
    core_decision = r1.get("core_decision", {}) if isinstance(r1.get("core_decision"), dict) else {}

    options = r2.get("options", []) if isinstance(r2.get("options"), list) else []
    selected_option = next(
        (x for x in options if isinstance(x, dict) and x.get("option_id") == selected_option_id),
        {},
    )

    topic = _first_nonempty(input_digest.get("topic"))
    core_expression = _first_nonempty(core_decision.get("core_expression"))
    option_title = _first_nonempty(selected_option.get("card_title"), selected_option.get("option_name"))
    option_summary = _first_nonempty(selected_option.get("card_summary"), selected_option.get("selection_reason"))

    script_title = _first_nonempty(
        base.get("script_title"),
        topic,
        option_title,
        "未命名主题",
    )
    base["script_title"] = script_title

    one_line_concept = _first_nonempty(
        base.get("one_line_concept"),
        core_expression,
        option_summary,
        option_title,
        script_title,
    )
    base["one_line_concept"] = one_line_concept or script_title


def _normalize_segment_groups(base: Dict[str, Any]) -> None:
    shots = base.get("shots")
    if not isinstance(shots, list) or len(shots) < 3:
        return

    ordered_shot_numbers: List[int] = []
    for shot in shots:
        if not isinstance(shot, dict):
            return
        shot_no = shot.get("shot_no")
        if not isinstance(shot_no, int) or shot_no <= 0:
            return
        ordered_shot_numbers.append(shot_no)

    expected = list(range(1, len(ordered_shot_numbers) + 1))
    if ordered_shot_numbers != expected:
        return

    existing_groups = base.get("segment_groups")
    existing_groups = existing_groups if isinstance(existing_groups, list) else []

    total_shots = len(ordered_shot_numbers)

    # 默认按 3 段切；如果原本就是 3~5 段，则优先保留段数
    segment_count = 3
    if 3 <= len(existing_groups) <= min(5, total_shots):
        segment_count = len(existing_groups)

    segment_count = min(segment_count, total_shots)

    default_names = ["开头片段", "主体片段", "结尾片段", "补充片段4", "补充片段5"]

    base_size = total_shots // segment_count
    remainder = total_shots % segment_count

    rebuilt_groups: List[Dict[str, Any]] = []
    cursor = 1

    for idx in range(segment_count):
        length = base_size + (1 if idx < remainder else 0)
        start_no = cursor
        end_no = cursor + length - 1

        source = (
            existing_groups[idx]
            if idx < len(existing_groups) and isinstance(existing_groups[idx], dict)
            else {}
        )

        segment_name = _first_nonempty(source.get("segment_name"), default_names[idx])
        segment_goal = _first_nonempty(
            source.get("segment_goal"),
            f"完成{segment_name}的内容推进",
        )
        segment_summary = _first_nonempty(
            source.get("segment_summary"),
            f"围绕{segment_name}完成该段内容展开",
        )

        rebuilt_groups.append(
            {
                "segment_id": f"segment_{idx + 1}",
                "segment_name": segment_name,
                "shot_range": f"镜头{start_no}-{end_no}" if start_no != end_no else f"镜头{start_no}",
                "related_shot_numbers": list(range(start_no, end_no + 1)),
                "segment_goal": segment_goal,
                "segment_summary": segment_summary,
            }
        )

        cursor = end_no + 1

    base["segment_groups"] = rebuilt_groups


def _normalize_dialogue_policy(base: Dict[str, Any], *, subtitle_only: bool) -> None:
    shots = base.get("shots")
    if not isinstance(shots, list):
        return

    segment_groups = base.get("segment_groups")
    segment_groups = segment_groups if isinstance(segment_groups, list) else []

    def find_segment_summary(shot_no: int) -> str:
        for segment in segment_groups:
            if not isinstance(segment, dict):
                continue
            related = segment.get("related_shot_numbers")
            if isinstance(related, list) and shot_no in related:
                return _first_nonempty(
                    segment.get("segment_summary"),
                    segment.get("segment_goal"),
                    segment.get("segment_name"),
                )
        return _first_nonempty(base.get("one_line_concept"), base.get("script_title"))

    for idx, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue

        dialogue = str(shot.get("dialogue", "")).strip()
        shot_no = shot.get("shot_no") if isinstance(shot.get("shot_no"), int) else idx + 1

        if subtitle_only:
            shot["dialogue"] = "无台词，以字幕呈现"
            continue

        if (not dialogue) or ("无台词" in dialogue) or ("以字幕呈现" in dialogue):
            summary = find_segment_summary(shot_no)

            if idx == 0:
                shot["dialogue"] = f"先自然带出主题：{summary}"
            elif idx == len(shots) - 1:
                shot["dialogue"] = f"最后收回重点，落到这段内容的核心感受：{summary}"
            else:
                shot["dialogue"] = f"继续补充真实细节，用自然口吻讲清这一段重点：{summary}"


def _sync_ai_instruction_display_from_base(
    base: Dict[str, Any],
    display: Any,
    *,
    subtitle_only: bool,
) -> None:
    if not isinstance(display, dict):
        return
    if display.get("display_type") != "segmented_ai_instructions":
        return

    segment_groups = base.get("segment_groups")
    if not isinstance(segment_groups, list) or not segment_groups:
        return

    cards = display.get("instruction_segments")
    cards = cards if isinstance(cards, list) else []

    rebuilt_cards: List[Dict[str, Any]] = []

    for idx, segment in enumerate(segment_groups):
        if not isinstance(segment, dict):
            return

        old_card = cards[idx] if idx < len(cards) and isinstance(cards[idx], dict) else {}

        old_instruction_text = _first_nonempty(old_card.get("instruction_text"))

        spoken_lines = old_card.get("spoken_lines")

        if subtitle_only:
            spoken_lines = ["无口播，以字幕呈现"]
        else:
            base_shots = base.get("shots", [])
            related_numbers = segment.get("related_shot_numbers", [])
            collected_lines: List[str] = []

            if isinstance(base_shots, list):
                for shot in base_shots:
                    if not isinstance(shot, dict):
                        continue
                    shot_no = shot.get("shot_no")
                    dialogue = str(shot.get("dialogue", "")).strip()
                    if (
                        isinstance(shot_no, int)
                        and shot_no in related_numbers
                        and dialogue
                        and "无台词" not in dialogue
                        and "以字幕呈现" not in dialogue
                    ):
                        collected_lines.append(dialogue)

            if collected_lines:
                spoken_lines = collected_lines[:3]
            else:
                spoken_lines = [
                    f"请用自然口吻讲清这段重点：{segment.get('segment_summary', segment.get('segment_goal', ''))}"
                ]

        if subtitle_only:
            instruction_text = _first_nonempty(
                old_instruction_text,
                f"请围绕{segment.get('segment_name', f'片段{idx + 1}')}生成对应AI指令，突出该片段重点画面、节奏、字幕信息与氛围表达。",
            )
        else:
            if ("无口播" in old_instruction_text) or ("以字幕呈现" in old_instruction_text) or ("纯字幕" in old_instruction_text):
                spoken_summary = "；".join(spoken_lines[:2])
                instruction_text = (
                    f"请围绕{segment.get('segment_name', f'片段{idx + 1}')}（{segment.get('shot_range', '')}）"
                    f"组织可直接执行的AI指令，重点突出该片段的画面推进、人物状态、环境细节与自然口播表达。"
                    f"口播重点：{spoken_summary}。字幕只做辅助强调，不要生成纯字幕+音乐版本。"
                )
            else:
                instruction_text = _first_nonempty(
                    old_instruction_text,
                    f"请围绕{segment.get('segment_name', f'片段{idx + 1}')}生成对应AI指令，突出该片段重点画面、节奏、口播与字幕配合。",
                )

        subtitle_focus = old_card.get("subtitle_focus")
        if not isinstance(subtitle_focus, list) or not subtitle_focus or any(
            not isinstance(x, str) or not x.strip() for x in subtitle_focus
        ):
            subtitle_focus = [segment.get("segment_name", f"片段{idx + 1}")]

        rebuilt_cards.append(
            {
                "segment_id": segment.get("segment_id", f"segment_{idx + 1}"),
                "segment_name": segment.get("segment_name", f"片段{idx + 1}"),
                "shot_range": segment.get("shot_range", ""),
                "related_shot_numbers": segment.get("related_shot_numbers", []),
                "segment_goal": segment.get("segment_goal", ""),
                "instruction_text": instruction_text,
                "spoken_lines": spoken_lines,
                "subtitle_focus": subtitle_focus[:3],
                "copy_enabled": True,
            }
        )

    display["instruction_segments"] = rebuilt_cards

    full_pkg = display.get("full_instruction_package")
    if not isinstance(full_pkg, dict):
        full_pkg = {}

    full_pkg["title"] = _first_nonempty(full_pkg.get("title"), "完整版AI指令")
    full_pkg["full_instruction_text"] = _first_nonempty(
        full_pkg.get("full_instruction_text"),
        "\n\n".join(card["instruction_text"] for card in rebuilt_cards if isinstance(card.get("instruction_text"), str)),
    )
    full_pkg["copy_enabled"] = True
    display["full_instruction_package"] = full_pkg


def _validate_api3_output(
    data: Dict[str, Any],
    *,
    result1_inner: Optional[Dict[str, Any]] = None,
    result2_inner: Optional[Dict[str, Any]] = None,
    selected_option_id: str = "",
    generation_route: str = "",
) -> None:
    if "error" in data:
        validate_json(data, ["error", "expected", "received"])
        return

    validate_json(data, ["result3"])
    validate_json(data["result3"], _RESULT3_INNER_KEYS)
    r3 = data["result3"]

    schema_version = _require_nonempty_str(r3, "schema_version", "result3")
    if schema_version != "v2.0":
        raise JSONParseError("result3.schema_version must be 'v2.0'")

    route = _require_nonempty_str(r3, "generation_route", "result3")
    if route not in _ALLOWED_ROUTES:
        raise JSONParseError("result3.generation_route must be 'ai_instruction' or 'pro_script'")
    if generation_route and route != generation_route:
        raise JSONParseError("result3.generation_route must equal requested generation_route")

    selected_id = _require_nonempty_str(r3, "selected_option_id", "result3")
    if selected_option_id and selected_id != selected_option_id:
        raise JSONParseError(
            f"result3.selected_option_id must equal requested selected_option_id {selected_option_id!r}"
        )
    _require_nonempty_str(r3, "selected_option_name", "result3")

    if r3.get("based_on_edited_result1") is not True:
        raise JSONParseError("result3.based_on_edited_result1 must be true")
    if r3.get("based_on_selected_option") is not True:
        raise JSONParseError("result3.based_on_selected_option must be true")

    source_trace = r3.get("source_trace")
    if not isinstance(source_trace, dict):
        raise JSONParseError("result3.source_trace must be an object")
    validate_json(
        source_trace,
        ["from_result1", "from_result2", "selected_option_id", "generation_route"],
    )
    if source_trace.get("from_result1") is not True or source_trace.get("from_result2") is not True:
        raise JSONParseError("source_trace.from_result1 and source_trace.from_result2 must both be true")
    if source_trace.get("selected_option_id") != selected_id:
        raise JSONParseError("source_trace.selected_option_id must equal result3.selected_option_id")
    if source_trace.get("generation_route") != route:
        raise JSONParseError("source_trace.generation_route must equal result3.generation_route")

    base = r3.get("base_storyboard_script")
    _validate_base_storyboard_script(base)

    display = r3.get("route_display_data")
    if route == "pro_script":
        _validate_pro_script_display(display)
    else:
        _validate_ai_instruction_display(display, base)


async def run_api3(
    client: AIClient,
    *,
    result1: Dict[str, Any],
    result2: Dict[str, Any],
    selected_option_id: str,
    generation_route: str,
    selected_option_data: Optional[Dict[str, Any]] = None,
    user_extra_request: str = "",
    edited_by_user_fields: Optional[list] = None,
    option_edited_by_user_fields: Optional[list] = None,
    language: str = "中文",
) -> Dict[str, Any]:
    if edited_by_user_fields is None:
        edited_by_user_fields = []
    if option_edited_by_user_fields is None:
        option_edited_by_user_fields = []
    if selected_option_data is None:
        selected_option_data = {}

    r1_inner = _unwrap_result1_blob(result1)
    r2_inner = _unwrap_result2_blob(result2)
    variables = {
        "result1": _dumps_json(r1_inner),
        "result2": _dumps_json(r2_inner),
        "selected_option_id": selected_option_id,
        "selected_option_data": _dumps_json(selected_option_data),
        "generation_route": generation_route,
        "user_extra_request": user_extra_request,
        "edited_by_user_fields": _dumps_json(edited_by_user_fields),
        "option_edited_by_user_fields": _dumps_json(option_edited_by_user_fields),
        "language": language,
    }
    system = _build_api3_system_prompt(generation_route, variables)
    raw = await client.request_chat(
        system, USER_FOOTER, response_format=JSON_OBJECT_FORMAT
    )
    data = parse_json_from_llm(raw)
    _fill_storyboard_required_fields(
        data,
        result1=result1,
        result2=result2,
        selected_option_id=selected_option_id,
    )
    subtitle_only = _is_explicit_subtitle_music(result1)

    if isinstance(data.get("result3"), dict):
        base = data["result3"].get("base_storyboard_script")
        if isinstance(base, dict):
            _normalize_segment_groups(base)
            _normalize_dialogue_policy(base, subtitle_only=subtitle_only)

            if data["result3"].get("generation_route") == "ai_instruction":
                _sync_ai_instruction_display_from_base(
                    base,
                    data["result3"].get("route_display_data"),
                    subtitle_only=subtitle_only,
                )

    _validate_api3_output(
        data,
        result1_inner=r1_inner,
        result2_inner=r2_inner,
        selected_option_id=selected_option_id,
        generation_route=generation_route,
    )
    return data