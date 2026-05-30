"""API2: result1 → result2 (three options).

与 API1 相同的 OpenAI 调用链：传入的 ``AIClient(api_stage="api2")`` 使用
``AsyncOpenAI.chat.completions``、``response_format={"type": "json_object"}``；
凭据读取 ``API_KEY`` / ``MODEL`` / ``API_BASE_URL``（缺省时回退 ``API1_*``）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set

from api.ai_client import AIClient
from utils.json_parser import JSONParseError, parse_json_from_llm, validate_json
from utils.prompt_loader import combine_prompts, load_prompt

JSON_OBJECT_FORMAT: Dict[str, str] = {"type": "json_object"}

USER_FOOTER = (
    "以上为完整任务说明与占位符已替换后的输入。"
    "请严格只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释性文字。"
)

_RESULT2_INNER_KEYS: List[str] = [
    "schema_version",
    "source_trace",
    "generation_route",
    "based_on_edited_result1",
    "platform_adaptation_summary",
    "global_constraints_summary",
    "options",
    "selection_hint",
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
_ALLOWED_OPTION_IDS: Sequence[str] = ("A", "B", "C")


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


def _build_api2_system_prompt(generation_route: str, variables: Dict[str, str]) -> str:
    if generation_route == "ai_instruction":
        route_file = "api2_route_ai_instruction.txt"
    elif generation_route == "pro_script":
        route_file = "api2_route_pro_script.txt"
    else:
        raise ValueError(
            "generation_route must be 'ai_instruction' or 'pro_script', "
            f"got {generation_route!r}"
        )
    base = load_prompt("api2", "api2_shared_base.txt")
    route = load_prompt("api2", route_file)
    combined = combine_prompts([base, route])
    return _apply_template(combined, variables)


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


# fallback 只负责提供“主题类型通用”的最低可用兜底，
# 不得默认假设当前主题一定是学习、咖啡厅、生活记录等单一主题。
# 所有 fallback 文案应优先保持中性，避免把模型拉回固定场景。

def _build_option_fallback_bundle(
    *,
    option_id: str,
    result1_inner: Dict[str, Any],
    generation_route: str,
) -> Dict[str, Any]:
    input_digest = result1_inner.get("input_digest", {}) if isinstance(result1_inner.get("input_digest"), dict) else {}
    core_decision = result1_inner.get("core_decision", {}) if isinstance(result1_inner.get("core_decision"), dict) else {}
    theme_analysis = result1_inner.get("theme_analysis", {}) if isinstance(result1_inner.get("theme_analysis"), dict) else {}
    goal_semantics = result1_inner.get("goal_semantics", {}) if isinstance(result1_inner.get("goal_semantics"), dict) else {}

    topic = _first_nonempty(
        input_digest.get("topic"),
        core_decision.get("core_expression"),
        "当前主题",
    )
    scene = _first_nonempty(theme_analysis.get("primary_scene"), "当前场景")
    action = _first_nonempty(theme_analysis.get("primary_action"), "当前行为")
    theme_signal = _first_nonempty(
        theme_analysis.get("theme_signal_summary"),
        f"围绕“{topic}”放大最值得被感受到的那一面。",
    )
    persona_traits = _clean_string_list(theme_analysis.get("possible_persona_traits"))
    primary_goal = _first_nonempty(core_decision.get("primary_goal"), "讲清主题")
    goal_focus = _first_nonempty(
        goal_semantics.get("goal_expression_focus"),
        f"表达重点围绕“{primary_goal}”展开。",
    )

    if generation_route == "ai_instruction":
        if "建立人设" in primary_goal:
            bundles = {
                "A": {
                    "card_title": f"在{scene}找到我的{action}方式",
                    "card_summary": f"这一版会先让人看到你在{scene}里最自然的{action}状态，重点把习惯、气质和个人表达讲出来。",
                    "highlight_tags": ["学习习惯", "个人状态", "轻松表达"],
                    "selection_reason": "适合先把自己的状态和习惯讲出来，让人更快记住你是怎样的人。",
                    "opening_approach": f"优先抛出你在{scene}里最有识别度的{action}状态。",
                    "core_structure_path": [
                        f"放大{scene}里的{action}状态",
                        "带出个人习惯和稳定节奏",
                        "收到更清晰的人设感上",
                    ],
                    "ending_approach": "收束到更稳定的人物气质和持续感上。",
                    "rhythm_style": "自然推进，重点放在状态识别度。",
                    "visual_organization": f"画面围绕{scene}细节、人物状态和动作习惯来组织。",
                },
                "B": {
                    "card_title": f"我的{scene}{action}仪式感",
                    "card_summary": f"这一版更会放大固定动作、环境细节和重复节奏，让人看到你是怎么把普通时刻过成自己的方式。",
                    "highlight_tags": ["仪式感", "行为规律", "生活节奏"],
                    "selection_reason": "适合把持续重复的那部分放大，更容易建立稳定标签感。",
                    "opening_approach": f"先让人看到你在{scene}里那些固定又自然的动作细节。",
                    "core_structure_path": [
                        f"抓住{scene}里的固定细节",
                        "带出重复动作和秩序感",
                        "收到更鲜明的个人标签上",
                    ],
                    "ending_approach": "收束到“这就是你的方式”这一感受上。",
                    "rhythm_style": "节奏平稳，强调重复动作带来的识别度。",
                    "visual_organization": "画面重点放在环境细节、固定动作和状态变化上。",
                },
                "C": {
                    "card_title": f"{scene}让我更能进入状态",
                    "card_summary": f"这一版会重点讲环境怎样影响你的状态与专注感，更适合把一种持续稳定的学习方式讲出来。",
                    "highlight_tags": ["场景影响", "专注状态", "学习方式"],
                    "selection_reason": "适合把环境和状态之间的关系讲清，让人更容易记住你的表达方式。",
                    "opening_approach": f"直接放大{scene}对状态产生影响的那一刻。",
                    "core_structure_path": [
                        f"点出{scene}里最影响状态的细节",
                        "带出你如何顺着环境进入节奏",
                        "收到更稳定的学习方式上",
                    ],
                    "ending_approach": "收束到环境如何塑造你的状态和方式上。",
                    "rhythm_style": "由场景细节带动状态推进，整体更沉浸。",
                    "visual_organization": f"画面围绕{scene}氛围、状态变化和人物专注感展开。",
                },
            }
        elif "引发共鸣" in primary_goal:
            bundles = {
                "A": {
                    "card_title": f"我在{scene}{action}的真实时刻",
                    "card_summary": f"这一版会先把最容易代入的那种真实状态讲出来，让人先觉得“我也有过这种时候”。",
                    "highlight_tags": ["真实体验", "情绪代入", "距离拉近"],
                    "selection_reason": "适合先用最真实的那部分打开，让观众更快代入你的感受。",
                    "opening_approach": f"直接抛出你在{scene}{action}时最真实的一瞬间。",
                    "core_structure_path": [
                        f"抛出{scene}{action}时最真实的状态",
                        "带出让人熟悉的细节感受",
                        "收到“很多人也会这样”上",
                    ],
                    "ending_approach": "收束到更容易让人代入的那种共同感受上。",
                    "rhythm_style": "先代入，再放大感受，整体更贴近人心。",
                    "visual_organization": f"画面围绕{scene}里的真实细节和人物状态展开。",
                },
                "B": {
                    "card_title": f"原来{scene}真的会影响状态",
                    "card_summary": f"这一版更会强调环境和情绪之间的关系，让人更容易把自己的经历代进去。",
                    "highlight_tags": ["环境影响", "状态变化", "共鸣入口"],
                    "selection_reason": "适合把感受怎么被环境放大的那部分讲出来，更容易引发共鸣。",
                    "opening_approach": f"先点出{scene}里最会影响状态的那个细节。",
                    "core_structure_path": [
                        f"点出{scene}里最影响状态的地方",
                        "带出感受如何被环境放大",
                        "收到更容易共鸣的体验上",
                    ],
                    "ending_approach": "收束到环境和内心状态之间的连接上。",
                    "rhythm_style": "先抛感受，再放大关系，整体更容易代入。",
                    "visual_organization": f"画面更重视{scene}氛围与人物感受之间的关系。",
                },
                "C": {
                    "card_title": f"这种{action}感受，很多人都懂",
                    "card_summary": f"这一版会把你从细节里感受到的东西慢慢带出来，更容易让人把自己的经历放进去。",
                    "highlight_tags": ["细节感受", "熟悉瞬间", "情感连接"],
                    "selection_reason": "适合把最细小但最容易让人点头的感受讲出来，距离会更近。",
                    "opening_approach": f"从一个最细小但最熟悉的{action}感受切入。",
                    "core_structure_path": [
                        f"抓住{action}时最细小的感受",
                        "带出这些细节为什么熟悉",
                        "收到更柔和的情感连接上",
                    ],
                    "ending_approach": "收束到更轻但更长尾的共鸣感上。",
                    "rhythm_style": "节奏更轻，重点放在细节带出的情绪连接。",
                    "visual_organization": "画面围绕小细节、小动作和情绪停顿来展开。",
                },
            }
        else:
            bundles = {
                "A": {
                    "card_title": f"先把{scene}里的状态讲出来",
                    "card_summary": "先把最容易被理解的那个状态或片段抛出来，会让主题更容易进入。",
                    "highlight_tags": ["状态切口", "真实表达", "主题清楚"],
                    "selection_reason": "适合先把最容易被理解的那部分讲出来。",
                    "opening_approach": f"优先抛出{scene}里最容易建立代入感的状态。",
                    "core_structure_path": [
                        f"抛出{scene}里的核心状态",
                        "带出这件事里最值得看的重点",
                        "收到更明确的主题表达上",
                    ],
                    "ending_approach": "收束到更清楚的主题感受上。",
                    "rhythm_style": "进入更直接，整体更清楚。",
                    "visual_organization": f"画面优先围绕{scene}和核心状态来组织。",
                },
                "B": {
                    "card_title": f"从{scene}细节里看{action}习惯",
                    "card_summary": "从更具体的细节和习惯入手，会让主题讲得更有画面感，也更容易看出重点。",
                    "highlight_tags": ["场景细节", "动作习惯", "画面感"],
                    "selection_reason": "适合把主题讲得更具体，让用户更容易看到细节差异。",
                    "opening_approach": f"从{scene}里最具体的那个细节开始进入。",
                    "core_structure_path": [
                        f"抓住{scene}里的关键细节",
                        f"带出{action}背后的习惯和节奏",
                        "收到更具体的表达重点上",
                    ],
                    "ending_approach": "收束到更具体的观察和感受上。",
                    "rhythm_style": "细节推动表达，节奏更稳。",
                    "visual_organization": "画面先细节、后状态，再落到整体感觉。",
                },
                "C": {
                    "card_title": f"把这件事收到更明确的感受上",
                    "card_summary": "会更集中地放大感受、判断或表达重心，让主题最后落到更清楚的结果上。",
                    "highlight_tags": ["感受重点", "表达收束", "情绪落点"],
                    "selection_reason": "适合把主题最后落到更明确的感受上，让记忆点更集中。",
                    "opening_approach": "先抛出最值得被感受到的那一面。",
                    "core_structure_path": [
                        "先点出最值得感受到的部分",
                        "再让重点逐步集中",
                        "最后收到更明确的情绪或判断上",
                    ],
                    "ending_approach": "收束到更清晰的感受落点上。",
                    "rhythm_style": "前面更轻，后面更聚焦。",
                    "visual_organization": "画面从状态和细节逐步集中到最终感受上。",
                },
            }
    else:
        bundles = {
            "A": {
                "card_title": f"先把{scene}里的状态讲出来",
                "card_summary": "先把最核心的内容状态或片段讲清，会让脚本更容易进入主题。",
                "highlight_tags": ["开头直接", "状态先行", "更好进入"],
                "selection_reason": "适合先把最核心的状态讲出来，让脚本更快进入主题。",
                "opening_approach": f"开头直接进入{scene}里的核心状态。",
                "core_structure_path": [
                    f"抛出{scene}里的状态",
                    "推进主体重点",
                    "收到更明确的表达结果上",
                ],
                "ending_approach": "收束到更清楚的结论或感受上。",
                "rhythm_style": "进入快、推进直。",
                "visual_organization": f"画面围绕{scene}里的核心动作和状态组织。",
            },
            "B": {
                "card_title": f"从细节里推进{action}节奏",
                "card_summary": "从动作、细节或关键片段推进主体，会让脚本结构更具体。",
                "highlight_tags": ["细节推进", "动作节奏", "结构具体"],
                "selection_reason": "适合把中段内容讲得更具体、更容易展开。",
                "opening_approach": f"先抓一个与{action}有关的细节进入。",
                "core_structure_path": [
                    "先抓细节建立画面感",
                    f"再推进{action}节奏",
                    "最后收到更完整的内容重点上",
                ],
                "ending_approach": "收束到更完整的表达闭环上。",
                "rhythm_style": "细节驱动推进，整体更稳。",
                "visual_organization": "画面从局部细节推进到完整内容。",
            },
            "C": {
                "card_title": f"把重点收到更清楚的感受上",
                "card_summary": "把脚本最后收到更清楚的感受、判断或记忆点上，会让整版更容易留下印象。",
                "highlight_tags": ["收束清楚", "记忆集中", "落点明确"],
                "selection_reason": "适合让脚本最后的记忆点更集中、更明确。",
                "opening_approach": "先点出最值得被记住的那一面。",
                "core_structure_path": [
                    "先点出内容里最值得记住的部分",
                    "再集中推进关键重点",
                    "最后收到更明确的落点上",
                ],
                "ending_approach": "收束到更明确的感受或判断上。",
                "rhythm_style": "前面铺垫更轻，后面更集中。",
                "visual_organization": "画面逐步收向更明确的重点和落点。",
            },
        }

    bundle = bundles.get(option_id, bundles["A"])

    if persona_traits and "建立人设" in primary_goal and generation_route == "ai_instruction":
        trait = persona_traits[0]
        bundle = dict(bundle)
        bundle["card_summary"] = f"{bundle['card_summary']} 重点会自然带出更明显的“{trait}”气质。"
        tags = list(bundle["highlight_tags"])
        if trait not in tags and len(tags) < 4:
            tags.append(trait)
        bundle["highlight_tags"] = tags[:4]

    if theme_signal and generation_route == "ai_instruction":
        bundle = dict(bundle)
        if "card_summary" in bundle and theme_signal not in bundle["card_summary"]:
            bundle["card_summary"] = f"{bundle['card_summary']} {theme_signal}"

    if goal_focus:
        bundle = dict(bundle)
        if "selection_reason" in bundle and goal_focus not in bundle["selection_reason"]:
            bundle["selection_reason"] = f"{bundle['selection_reason']} {goal_focus}"

    return bundle


def _fallback_core_structure_path(
    generation_route: str,
    title: str,
    summary: str,
    *,
    fallback_bundle: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if isinstance(fallback_bundle, dict):
        bundle_path = _clean_string_list(fallback_bundle.get("core_structure_path"))
        if len(bundle_path) >= 3:
            return bundle_path[:5]

    if generation_route == "ai_instruction":
        return [
            "放大最有代入感的那个状态",
            "带出更具体的细节和重点",
            "收到更明确的感受或标签上",
        ]
    return [
        "先点出最核心的内容切口",
        "推进主体里的关键重点",
        "收束到更明确的表达落点上",
    ]


def _fallback_material_usage(material_status: str, generation_route: str) -> str:
    status = (material_status or "").strip()
    if status == "has_material":
        return "优先利用现有素材组织主体内容，缺少部分再补拍或补生成。"
    if status == "partial_material":
        return "保留现有可用素材，主体按轻补拍或低成本补充来完成。"
    if generation_route == "ai_instruction":
        return "按低素材依赖思路组织，优先口播、基础场景和可补生成片段。"
    return "按低素材依赖思路组织，优先口播、基础场景和简单补拍。"

def _normalize_expansion_anchor(
    anchor: Any,
    *,
    option: Dict[str, Any],
    result1_inner: Dict[str, Any],
    result2_obj: Dict[str, Any],
    generation_route: str,
    fallback_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(anchor, dict):
        anchor = {}

    input_digest = result1_inner.get("input_digest", {}) if isinstance(result1_inner.get("input_digest"), dict) else {}
    core_decision = result1_inner.get("core_decision", {}) if isinstance(result1_inner.get("core_decision"), dict) else {}
    platform_adaptation = result1_inner.get("platform_adaptation", {}) if isinstance(result1_inner.get("platform_adaptation"), dict) else {}
    material_analysis = result1_inner.get("material_analysis", {}) if isinstance(result1_inner.get("material_analysis"), dict) else {}
    media_style = result1_inner.get("media_style_inference", {}) if isinstance(result1_inner.get("media_style_inference"), dict) else {}

    platform_summary = result2_obj.get("platform_adaptation_summary", {}) if isinstance(result2_obj.get("platform_adaptation_summary"), dict) else {}
    global_summary = result2_obj.get("global_constraints_summary", {}) if isinstance(result2_obj.get("global_constraints_summary"), dict) else {}

    bundle = fallback_bundle if isinstance(fallback_bundle, dict) else {}

    card_title = _first_nonempty(option.get("card_title"))
    card_summary = _first_nonempty(option.get("card_summary"))
    selection_reason = _first_nonempty(option.get("selection_reason"))
    topic = _first_nonempty(
        input_digest.get("topic"),
        core_decision.get("core_expression"),
        global_summary.get("core_expression"),
        card_title,
    )

    material_status = _first_nonempty(
        material_analysis.get("material_status"),
        global_summary.get("material_status"),
    )

    anchor["opening_approach"] = _first_nonempty(
        anchor.get("opening_approach"),
        bundle.get("opening_approach"),
        card_summary,
        f"优先抛出与“{topic}”最有代入感的那个切口。",
    )

    core_path = _clean_string_list(anchor.get("core_structure_path"))
    if len(core_path) < 3:
        core_path = _fallback_core_structure_path(
            generation_route,
            card_title,
            card_summary,
            fallback_bundle=bundle,
        )
    anchor["core_structure_path"] = core_path[:5]

    anchor["ending_approach"] = _first_nonempty(
        anchor.get("ending_approach"),
        bundle.get("ending_approach"),
        selection_reason,
        "收束到更明确的感受、判断或人物标签上。",
    )

    anchor["rhythm_style"] = _first_nonempty(
        anchor.get("rhythm_style"),
        bundle.get("rhythm_style"),
        platform_adaptation.get("recommended_rhythm_style"),
        platform_summary.get("platform_rhythm_preference"),
        "节奏清楚，重点逐步集中。",
    )

    anchor["visual_organization"] = _first_nonempty(
        anchor.get("visual_organization"),
        bundle.get("visual_organization"),
        "画面围绕最值得被看见的状态、细节和重点来组织。",
    )

    anchor["dialogue_tone"] = _first_nonempty(
        anchor.get("dialogue_tone"),
        platform_adaptation.get("recommended_expression_style"),
        input_digest.get("narration_mode"),
        "自然、清楚、便于表达。",
    )

    anchor["material_usage_plan"] = _first_nonempty(
        anchor.get("material_usage_plan"),
        _fallback_material_usage(material_status, generation_route),
    )

    anchor["platform_fit_focus"] = _first_nonempty(
        anchor.get("platform_fit_focus"),
        platform_summary.get("platform_expression_note"),
        platform_summary.get("platform_content_style"),
        platform_adaptation.get("platform_fit_reason"),
        f"表达方式与{_first_nonempty(input_digest.get('target_platform'), platform_summary.get('target_platform'), '当前平台')}保持一致。",
    )

    anchor["media_fit_focus"] = _first_nonempty(
        anchor.get("media_fit_focus"),
        global_summary.get("recommended_media_type"),
        media_style.get("recommended_media_type"),
        "媒介表达与当前内容方向保持一致。",
    )

    anchor["style_fit_focus"] = _first_nonempty(
        anchor.get("style_fit_focus"),
        global_summary.get("main_style_note"),
        (media_style.get("visual_style") or {}).get("style_summary") if isinstance(media_style.get("visual_style"), dict) else "",
        "整体风格保持统一、清楚、自然。",
    )

    return anchor


def _normalize_option(
    option: Any,
    *,
    option_id: str,
    result1_inner: Dict[str, Any],
    result2_obj: Dict[str, Any],
    generation_route: str,
) -> Dict[str, Any]:
    if not isinstance(option, dict):
        option = {}

    bundle = _build_option_fallback_bundle(
        option_id=option_id,
        result1_inner=result1_inner,
        generation_route=generation_route,
    )

    option["option_id"] = option_id
    option["card_title"] = _first_nonempty(
        option.get("card_title"),
        bundle.get("card_title"),
    )
    option["card_summary"] = _first_nonempty(
        option.get("card_summary"),
        bundle.get("card_summary"),
    )

    tags = _clean_string_list(option.get("highlight_tags"))
    default_tags = _clean_string_list(bundle.get("highlight_tags"))
    for tag in default_tags:
        if len(tags) >= 4:
            break
        if tag not in tags:
            tags.append(tag)
    option["highlight_tags"] = tags[:4] if tags else default_tags[:4]

    option["selection_reason"] = _first_nonempty(
        option.get("selection_reason"),
        bundle.get("selection_reason"),
    )

    option["expansion_anchor"] = _normalize_expansion_anchor(
        option.get("expansion_anchor"),
        option=option,
        result1_inner=result1_inner,
        result2_obj=result2_obj,
        generation_route=generation_route,
        fallback_bundle=bundle,
    )
    return option
def _normalize_api2_output(
    data: Dict[str, Any],
    *,
    result1_inner: Dict[str, Any],
    generation_route: str,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}

    if not isinstance(data.get("result2"), dict):
        data["result2"] = {}

    r2 = data["result2"]

    r2["schema_version"] = _first_nonempty(r2.get("schema_version"), "v2.0")
    r2["generation_route"] = generation_route
    r2["based_on_edited_result1"] = True

    if not isinstance(r2.get("source_trace"), dict):
        r2["source_trace"] = {}
    r2["source_trace"]["from_result1"] = True
    r2["source_trace"]["generation_route"] = generation_route

    input_digest = result1_inner.get("input_digest", {}) if isinstance(result1_inner.get("input_digest"), dict) else {}
    core_decision = result1_inner.get("core_decision", {}) if isinstance(result1_inner.get("core_decision"), dict) else {}
    platform_adaptation = result1_inner.get("platform_adaptation", {}) if isinstance(result1_inner.get("platform_adaptation"), dict) else {}
    audience_focus = result1_inner.get("audience_focus", {}) if isinstance(result1_inner.get("audience_focus"), dict) else {}
    presentation_decision = result1_inner.get("presentation_decision", {}) if isinstance(result1_inner.get("presentation_decision"), dict) else {}
    narration_decision = result1_inner.get("narration_decision", {}) if isinstance(result1_inner.get("narration_decision"), dict) else {}
    material_analysis = result1_inner.get("material_analysis", {}) if isinstance(result1_inner.get("material_analysis"), dict) else {}
    media_style = result1_inner.get("media_style_inference", {}) if isinstance(result1_inner.get("media_style_inference"), dict) else {}

    if not isinstance(r2.get("platform_adaptation_summary"), dict):
        r2["platform_adaptation_summary"] = {}
    pas = r2["platform_adaptation_summary"]
    pas["target_platform"] = _first_nonempty(pas.get("target_platform"), input_digest.get("target_platform"), platform_adaptation.get("target_platform"), "小红书")
    pas["platform_content_style"] = _first_nonempty(pas.get("platform_content_style"), platform_adaptation.get("platform_content_style"), "真实、清楚、贴近平台表达习惯。")
    pas["platform_opening_preference"] = _first_nonempty(pas.get("platform_opening_preference"), platform_adaptation.get("recommended_opening_style"), "开头尽快建立主题和代入感。")
    pas["platform_rhythm_preference"] = _first_nonempty(pas.get("platform_rhythm_preference"), platform_adaptation.get("recommended_rhythm_style"), "节奏清楚，不拖沓。")
    pas["platform_expression_note"] = _first_nonempty(pas.get("platform_expression_note"), platform_adaptation.get("recommended_expression_style"), "表达自然，符合平台内容感受。")

    if not isinstance(r2.get("global_constraints_summary"), dict):
        r2["global_constraints_summary"] = {}
    gcs = r2["global_constraints_summary"]
    gcs["core_expression"] = _first_nonempty(gcs.get("core_expression"), core_decision.get("core_expression"), input_digest.get("topic"), "当前主题表达")
    gcs["primary_goal"] = _first_nonempty(gcs.get("primary_goal"), core_decision.get("primary_goal"), "讲清主题")
    gcs["secondary_goal"] = _first_nonempty(gcs.get("secondary_goal"), core_decision.get("secondary_goal"), "增强记忆点")
    gcs["duration_bucket"] = _first_nonempty(gcs.get("duration_bucket"), input_digest.get("duration_preference"), "短视频时长")
    gcs["primary_audience"] = _first_nonempty(gcs.get("primary_audience"), audience_focus.get("primary_audience"), input_digest.get("target_audience"), "目标观众")
    gcs["primary_presentation_mode"] = _first_nonempty(gcs.get("primary_presentation_mode"), input_digest.get("presentation_mode"), presentation_decision.get("recommended_presentation_mode"), "常规出镜")
    gcs["primary_narration_mode"] = _first_nonempty(gcs.get("primary_narration_mode"), input_digest.get("narration_mode"), narration_decision.get("recommended_narration_mode"), "自然表达")
    gcs["material_status"] = _first_nonempty(gcs.get("material_status"), material_analysis.get("material_status"), "no_material")
    gcs["recommended_media_type"] = _first_nonempty(gcs.get("recommended_media_type"), media_style.get("recommended_media_type"), "真人实拍")
    gcs["main_style_note"] = _first_nonempty(
        gcs.get("main_style_note"),
        ((media_style.get("visual_style") or {}) if isinstance(media_style.get("visual_style"), dict) else {}).get("style_summary"),
        "整体风格保持统一、自然、清楚。",
    )
    main_risks = _clean_string_list(gcs.get("main_risks"))
    if not main_risks:
        main_risks = ["避免内容与主题脱节"]
    gcs["main_risks"] = main_risks

    raw_options = r2.get("options")
    if not isinstance(raw_options, list):
        raw_options = []
    normalized_options: List[Dict[str, Any]] = []
    for idx, option_id in enumerate(("A", "B", "C")):
        option = raw_options[idx] if idx < len(raw_options) else {}
        normalized_options.append(
            _normalize_option(
                option,
                option_id=option_id,
                result1_inner=result1_inner,
                result2_obj=r2,
                generation_route=generation_route,
            )
        )
    r2["options"] = normalized_options

    if not isinstance(r2.get("selection_hint"), dict):
        r2["selection_hint"] = {}
    hint = r2["selection_hint"]
    hint["if_user_wants_fastest_grasp"] = _first_nonempty(hint.get("if_user_wants_fastest_grasp"), "先看 A，更容易快速进入主题。")
    hint["if_user_wants_clearest_delivery"] = _first_nonempty(hint.get("if_user_wants_clearest_delivery"), "先看 B，更适合把内容讲清楚。")
    hint["if_user_wants_strongest_style_or_memory"] = _first_nonempty(hint.get("if_user_wants_strongest_style_or_memory"), "先看 C，更容易形成记忆点。")

    return data


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


def _require_string_list(
    obj: Dict[str, Any],
    key: str,
    ctx: str,
    *,
    min_len: int = 1,
    max_len: Optional[int] = None,
) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list):
        raise JSONParseError(f"{ctx}.{key} must be a list")
    if len(value) < min_len:
        raise JSONParseError(f"{ctx}.{key} must contain at least {min_len} items")
    if max_len is not None and len(value) > max_len:
        raise JSONParseError(f"{ctx}.{key} must contain at most {max_len} items")
    cleaned: List[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise JSONParseError(f"{ctx}.{key}[{idx}] must be a non-empty string")
        cleaned.append(item.strip())
    return cleaned


def _normalize_compare_text(text: str) -> str:
    cleaned = []
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
    return "".join(cleaned)


def _char_ngrams(text: str, n: int = 2) -> Set[str]:
    normalized = _normalize_compare_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _text_similarity(a: str, b: str) -> float:
    grams_a = _char_ngrams(a)
    grams_b = _char_ngrams(b)
    if not grams_a or not grams_b:
        return 0.0
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


def _tag_overlap_ratio(tags_a: List[str], tags_b: List[str]) -> float:
    set_a = {t.strip() for t in tags_a if isinstance(t, str) and t.strip()}
    set_b = {t.strip() for t in tags_b if isinstance(t, str) and t.strip()}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _looks_template_like(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False

    banned_phrases = [
        "这一版会围绕",
        "并给出更清楚的展开路径",
        "适合先把主题讲清",
        "更适合当前输入",
        "当前输入条件",
        "当前条件下",
        "先把主题讲清",
        "围绕当前主题组织内容",
    ]
    return any(phrase in normalized for phrase in banned_phrases)


def _validate_platform_adaptation_summary(summary: Any) -> None:
    if not isinstance(summary, dict):
        raise JSONParseError("result2.platform_adaptation_summary must be an object")
    validate_json(
        summary,
        [
            "target_platform",
            "platform_content_style",
            "platform_opening_preference",
            "platform_rhythm_preference",
            "platform_expression_note",
        ],
    )
    platform = _require_nonempty_str(summary, "target_platform", "result2.platform_adaptation_summary")
    if platform not in _ALLOWED_PLATFORMS:
        raise JSONParseError(
            f"result2.platform_adaptation_summary.target_platform must be one of {sorted(_ALLOWED_PLATFORMS)}"
        )
    _require_nonempty_str(summary, "platform_content_style", "result2.platform_adaptation_summary")
    _require_nonempty_str(summary, "platform_opening_preference", "result2.platform_adaptation_summary")
    _require_nonempty_str(summary, "platform_rhythm_preference", "result2.platform_adaptation_summary")
    _require_nonempty_str(summary, "platform_expression_note", "result2.platform_adaptation_summary")


def _validate_global_constraints_summary(summary: Any) -> None:
    if not isinstance(summary, dict):
        raise JSONParseError("result2.global_constraints_summary must be an object")
    validate_json(
        summary,
        [
            "core_expression",
            "primary_goal",
            "secondary_goal",
            "duration_bucket",
            "primary_audience",
            "primary_presentation_mode",
            "primary_narration_mode",
            "material_status",
            "recommended_media_type",
            "main_style_note",
            "main_risks",
        ],
    )
    for key in [
        "core_expression",
        "primary_goal",
        "secondary_goal",
        "duration_bucket",
        "primary_audience",
        "primary_presentation_mode",
        "primary_narration_mode",
        "material_status",
        "main_style_note",
    ]:
        _require_nonempty_str(summary, key, "result2.global_constraints_summary")

    media_type = _require_nonempty_str(
        summary, "recommended_media_type", "result2.global_constraints_summary"
    )
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise JSONParseError(
            f"result2.global_constraints_summary.recommended_media_type must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        )

    main_risks = summary.get("main_risks")
    if not isinstance(main_risks, list):
        raise JSONParseError("result2.global_constraints_summary.main_risks must be a list")
    for idx, item in enumerate(main_risks):
        if not isinstance(item, str):
            raise JSONParseError(
                f"result2.global_constraints_summary.main_risks[{idx}] must be a string"
            )


def _validate_expansion_anchor(anchor: Any, ctx: str) -> None:
    if not isinstance(anchor, dict):
        raise JSONParseError(f"{ctx} must be an object")
    validate_json(
        anchor,
        [
            "opening_approach",
            "core_structure_path",
            "ending_approach",
            "rhythm_style",
            "visual_organization",
            "dialogue_tone",
            "material_usage_plan",
            "platform_fit_focus",
            "media_fit_focus",
            "style_fit_focus",
        ],
    )

    for key in [
        "opening_approach",
        "ending_approach",
        "rhythm_style",
        "visual_organization",
        "dialogue_tone",
        "material_usage_plan",
        "platform_fit_focus",
        "media_fit_focus",
        "style_fit_focus",
    ]:
        _require_nonempty_str(anchor, key, ctx)

    _require_string_list(anchor, "core_structure_path", ctx, min_len=3, max_len=5)


def _validate_options(options: Any) -> None:
    if not isinstance(options, list) or len(options) != 3:
        raise JSONParseError("result2.options must be a list of exactly 3 items")

    seen_ids: List[str] = []
    seen_titles: Set[str] = set()

    summary_texts: List[str] = []
    reason_texts: List[str] = []
    tag_groups: List[List[str]] = []

    for idx, option in enumerate(options):
        ctx = f"result2.options[{idx}]"
        if not isinstance(option, dict):
            raise JSONParseError(f"{ctx} must be an object")
        validate_json(
            option,
            [
                "option_id",
                "card_title",
                "card_summary",
                "highlight_tags",
                "selection_reason",
                "expansion_anchor",
            ],
        )

        option_id = _require_nonempty_str(option, "option_id", ctx)
        seen_ids.append(option_id)
        if option_id not in _ALLOWED_OPTION_IDS:
            raise JSONParseError(f"{ctx}.option_id must be one of {_ALLOWED_OPTION_IDS}")

        title = _require_nonempty_str(option, "card_title", ctx)
        if title in seen_titles:
            raise JSONParseError("result2.options.card_title must be distinct across all options")
        seen_titles.add(title)

        summary = _require_nonempty_str(option, "card_summary", ctx)
        if len(summary) < 12:
            raise JSONParseError(f"{ctx}.card_summary is too short; need at least 12 characters")
        if _looks_template_like(summary):
            raise JSONParseError(f"{ctx}.card_summary is too template-like; rewrite in more natural user-facing language")
        summary_texts.append(summary)

        tags = _require_string_list(option, "highlight_tags", ctx, min_len=3, max_len=4)
        if len(set(tags)) != len(tags):
            raise JSONParseError(f"{ctx}.highlight_tags must not contain duplicates")
        tag_groups.append(tags)

        reason = _require_nonempty_str(option, "selection_reason", ctx)
        if len(reason) < 8:
            raise JSONParseError(f"{ctx}.selection_reason is too short; need at least 8 characters")
        if _looks_template_like(reason):
            raise JSONParseError(f"{ctx}.selection_reason is too template-like; rewrite in more natural user-facing language")
        reason_texts.append(reason)

        _validate_expansion_anchor(option.get("expansion_anchor"), f"{ctx}.expansion_anchor")

    if seen_ids != list(_ALLOWED_OPTION_IDS):
        raise JSONParseError("result2.options.option_id must appear in exact order: A, B, C")

    for i in range(len(summary_texts)):
        for j in range(i + 1, len(summary_texts)):
            summary_similarity = _text_similarity(summary_texts[i], summary_texts[j])
            if summary_similarity >= 0.72:
                raise JSONParseError(
                    f"result2.options[{i}].card_summary and result2.options[{j}].card_summary are too similar; need clearer differentiation"
                )

            reason_similarity = _text_similarity(reason_texts[i], reason_texts[j])
            if reason_similarity >= 0.76:
                raise JSONParseError(
                    f"result2.options[{i}].selection_reason and result2.options[{j}].selection_reason are too similar; need clearer differentiation"
                )

            tag_overlap = _tag_overlap_ratio(tag_groups[i], tag_groups[j])
            if tag_overlap > 0.75:
                raise JSONParseError(
                    f"result2.options[{i}].highlight_tags and result2.options[{j}].highlight_tags overlap too much; need more distinctive tags"
                )


def _validate_selection_hint(selection_hint: Any) -> None:
    if not isinstance(selection_hint, dict):
        raise JSONParseError("result2.selection_hint must be an object")
    validate_json(
        selection_hint,
        [
            "if_user_wants_fastest_grasp",
            "if_user_wants_clearest_delivery",
            "if_user_wants_strongest_style_or_memory",
        ],
    )
    for key in [
        "if_user_wants_fastest_grasp",
        "if_user_wants_clearest_delivery",
        "if_user_wants_strongest_style_or_memory",
    ]:
        _require_nonempty_str(selection_hint, key, "result2.selection_hint")


def _validate_api2_output(
    data: Dict[str, Any],
    *,
    generation_route: str = "",
) -> None:
    if "error" in data:
        validate_json(data, ["error", "expected", "received"])
        return

    validate_json(data, ["result2"])
    validate_json(data["result2"], _RESULT2_INNER_KEYS)
    r2 = data["result2"]

    schema_version = _require_nonempty_str(r2, "schema_version", "result2")
    if schema_version != "v2.0":
        raise JSONParseError("result2.schema_version must be 'v2.0'")

    source_trace = r2.get("source_trace")
    if not isinstance(source_trace, dict):
        raise JSONParseError("result2.source_trace must be an object")
    validate_json(source_trace, ["from_result1", "generation_route"])
    if source_trace.get("from_result1") is not True:
        raise JSONParseError("result2.source_trace.from_result1 must be true")

    route = _require_nonempty_str(r2, "generation_route", "result2")
    if route not in _ALLOWED_ROUTES:
        raise JSONParseError("result2.generation_route must be 'ai_instruction' or 'pro_script'")
    if generation_route and route != generation_route:
        raise JSONParseError("result2.generation_route must equal requested generation_route")
    if source_trace.get("generation_route") != route:
        raise JSONParseError("result2.source_trace.generation_route must equal result2.generation_route")

    if r2.get("based_on_edited_result1") is not True:
        raise JSONParseError("result2.based_on_edited_result1 must be true")

    _validate_platform_adaptation_summary(r2.get("platform_adaptation_summary"))
    _validate_global_constraints_summary(r2.get("global_constraints_summary"))
    _validate_options(r2.get("options"))
    _validate_selection_hint(r2.get("selection_hint"))


async def run_api2(
    client: AIClient,
    *,
    result1: Dict[str, Any],
    generation_route: str,
    user_extra_request: str = "",
    edited_by_user_fields: Optional[list] = None,
    language: str = "中文",
) -> Dict[str, Any]:
    if edited_by_user_fields is None:
        edited_by_user_fields = []

    inner = _unwrap_result1_blob(result1)
    variables = {
        "result1": _dumps_json(inner),
        "generation_route": generation_route,
        "user_extra_request": user_extra_request,
        "edited_by_user_fields": _dumps_json(edited_by_user_fields),
        "language": language,
    }
    system = _build_api2_system_prompt(generation_route, variables)
    raw = await client.request_chat(
        system, USER_FOOTER, response_format=JSON_OBJECT_FORMAT
    )
    data = parse_json_from_llm(raw)
    data = _normalize_api2_output(
        data,
        result1_inner=inner,
        generation_route=generation_route,
    )
    _validate_api2_output(data, generation_route=generation_route)
    return data