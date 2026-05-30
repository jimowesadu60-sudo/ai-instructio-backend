"""CLI entry: run API1 / API2 / API3 handlers.

API1–API3 使用同一套 OpenAI 凭据（``.env`` 中 ``API_KEY`` / ``MODEL`` /
``API_BASE_URL``，或 ``API1_*`` 回退）。未指定 ``api_stage`` 的 ``AIClient`` 仍读 ``AI_*``。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from api import AIClient, AIClientError, run_api1, run_api2, run_api3
from utils.json_parser import JSONParseError

load_dotenv()


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _load_json_object(path: Path, *, arg_name: str) -> Dict[str, Any]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{arg_name} file must contain a JSON object")
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_json_array_arg(raw: str, *, arg_name: str) -> List[Any]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be a valid JSON array string") from exc
    if not isinstance(data, list):
        raise ValueError(f"{arg_name} must be a JSON array")
    return data


def _unwrap_result2(blob: Dict[str, Any]) -> Dict[str, Any]:
    if "result2" in blob and isinstance(blob["result2"], dict):
        return blob["result2"]
    return blob


def _resolve_selected_option_data(
    *,
    result2_blob: Dict[str, Any],
    option_id: str,
    selected_option_data_path: Optional[str],
) -> Dict[str, Any]:
    if selected_option_data_path:
        return _load_json_object(
            Path(selected_option_data_path),
            arg_name="--selected-option-data",
        )

    result2_inner = _unwrap_result2(result2_blob)
    options = result2_inner.get("options", [])
    if isinstance(options, list):
        for item in options:
            if isinstance(item, dict) and item.get("option_id") == option_id:
                return item
    return {}


async def _cmd_api1(args: argparse.Namespace) -> None:
    data = await run_api1(
        topic=args.topic,
        goals=args.goals or "",
        platforms=args.platforms or "",
        duration=args.duration or "",
        audiences=args.audiences or "",
        presentation_mode=args.presentation_mode or "",
        narration_mode=args.narration_mode or "",
        materials=args.materials or "",
        extra_notes=args.extra_notes or "",
        image_paths=args.image,
    )
    if args.out:
        _write_json(Path(args.out), data)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


async def _cmd_api2(args: argparse.Namespace) -> None:
    client = AIClient(api_stage="api2")
    result1 = _load_json_object(Path(args.result1), arg_name="--result1")
    edited = _parse_json_array_arg(args.edited_fields, arg_name="--edited-fields")

    data = await run_api2(
        client,
        result1=result1,
        generation_route=args.route,
        user_extra_request=args.user_extra_request or "",
        edited_by_user_fields=edited,
        language=args.language or "中文",
    )
    if args.out:
        _write_json(Path(args.out), data)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


async def _cmd_api3(args: argparse.Namespace) -> None:
    client = AIClient(api_stage="api3")
    result1 = _load_json_object(Path(args.result1), arg_name="--result1")
    result2 = _load_json_object(Path(args.result2), arg_name="--result2")
    edited = _parse_json_array_arg(args.edited_fields, arg_name="--edited-fields")
    opt_edited = _parse_json_array_arg(
        args.option_edited_fields,
        arg_name="--option-edited-fields",
    )
    selected_data = _resolve_selected_option_data(
        result2_blob=result2,
        option_id=args.option_id,
        selected_option_data_path=args.selected_option_data,
    )

    data = await run_api3(
        client,
        result1=result1,
        result2=result2,
        selected_option_id=args.option_id,
        generation_route=args.route,
        selected_option_data=selected_data,
        user_extra_request=args.user_extra_request or "",
        edited_by_user_fields=edited,
        option_edited_by_user_fields=opt_edited,
        language=args.language or "中文",
    )
    if args.out:
        _write_json(Path(args.out), data)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


async def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Run api1 → api2 → api3; write outputs and optionally an aggregate payload.

    Route 只通过 ``generation_route`` / ``args.route`` 传递；
    不把 route 字符串塞进 ``extra_notes`` 或 ``user_extra_request``。
    """
    generation_route = args.route
    selected_option_id = args.option_id

    edited = _parse_json_array_arg(args.edited_fields, arg_name="--edited-fields")
    opt_edited = _parse_json_array_arg(
        args.option_edited_fields,
        arg_name="--option-edited-fields",
    )

    # 1) api1 -> outputs/api1/result1.json
    result1 = await run_api1(
        topic=args.topic,
        goals=args.goals or "",
        platforms=args.platforms or "",
        duration=args.duration or "",
        audiences=args.audiences or "",
        presentation_mode=args.presentation_mode or "",
        narration_mode=args.narration_mode or "",
        materials=args.materials or "",
        extra_notes=args.extra_notes or "",
        image_paths=args.image,
    )
    _write_json(Path("outputs/api1/result1.json"), result1)
    print("Wrote outputs/api1/result1.json")

    # 2) api2 -> outputs/api2/result2.json
    client2 = AIClient(api_stage="api2")
    result2 = await run_api2(
        client2,
        result1=result1,
        generation_route=generation_route,
        user_extra_request=args.user_extra_request or "",
        edited_by_user_fields=edited,
        language=args.language or "中文",
    )
    _write_json(Path("outputs/api2/result2.json"), result2)
    print("Wrote outputs/api2/result2.json")

    # 3) api3 -> outputs/api3/result3.json
    selected_data = _resolve_selected_option_data(
        result2_blob=result2,
        option_id=selected_option_id,
        selected_option_data_path=args.selected_option_data,
    )

    client3 = AIClient(api_stage="api3")
    result3 = await run_api3(
        client3,
        result1=result1,
        result2=result2,
        selected_option_id=selected_option_id,
        generation_route=generation_route,
        selected_option_data=selected_data,
        user_extra_request=args.user_extra_request or "",
        edited_by_user_fields=edited,
        option_edited_by_user_fields=opt_edited,
        language=args.language or "中文",
    )
    _write_json(Path("outputs/api3/result3.json"), result3)
    print("Wrote outputs/api3/result3.json")

    pipeline_result: Dict[str, Any] = {
        "schema_version": "v2.0",
        "generation_route": generation_route,
        "selected_option_id": selected_option_id,
        "result1": result1,
        "result2": result2,
        "result3": result3,
    }

    if args.out:
        _write_json(Path(args.out), pipeline_result)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(pipeline_result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-video pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("api1", help="Input analysis → result1")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--goals", default="")
    p1.add_argument("--platforms", default="")
    p1.add_argument("--duration", default="")
    p1.add_argument("--audiences", default="")
    p1.add_argument("--presentation-mode", default="")
    p1.add_argument("--narration-mode", default="")
    p1.add_argument("--materials", default="")
    p1.add_argument("--extra-notes", default="")
    p1.add_argument(
        "--image",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "素材图片路径，可重复传入多张；读取本地文件后以 gpt-4o-mini 多模态分析，"
            "结果写入 result1.material_analysis（输出 JSON 结构与无图时一致）"
        ),
    )
    p1.add_argument("--out", help="Write JSON to this path")
    p1.set_defaults(func=_cmd_api1)

    p2 = sub.add_parser("api2", help="result1 → result2 (three options)")
    p2.add_argument(
        "--input",
        "--result1",
        dest="result1",
        required=True,
        metavar="PATH",
        help="api1 输出的 result1 JSON（--input 与 --result1 等价）",
    )
    p2.add_argument(
        "--route",
        required=True,
        choices=("ai_instruction", "pro_script"),
    )
    p2.add_argument("--user-extra-request", default="")
    p2.add_argument(
        "--edited-fields",
        default="[]",
        help='JSON array, e.g. \'["core_decision.primary_goal"]\'',
    )
    p2.add_argument("--language", default="中文")
    p2.add_argument(
        "--out",
        metavar="PATH",
        help="写入 result2 JSON 的路径；若文件已存在则直接覆盖",
    )
    p2.set_defaults(func=_cmd_api2)

    p3 = sub.add_parser("api3", help="Expand selected option → result3")
    p3.add_argument("--result1", required=True, help="JSON file (api1 output)")
    p3.add_argument("--result2", required=True, help="JSON file (api2 output)")
    p3.add_argument(
        "--option-id",
        required=True,
        choices=("A", "B", "C"),
        help="Selected option id",
    )
    p3.add_argument(
        "--route",
        required=True,
        choices=("ai_instruction", "pro_script"),
    )
    p3.add_argument(
        "--selected-option-data",
        help="Optional JSON file with the chosen option object; if omitted, it is auto-resolved from result2",
    )
    p3.add_argument("--user-extra-request", default="")
    p3.add_argument("--edited-fields", default="[]")
    p3.add_argument("--option-edited-fields", default="[]")
    p3.add_argument("--language", default="中文")
    p3.add_argument("--out", help="Write JSON to this path")
    p3.set_defaults(func=_cmd_api3)

    p4 = sub.add_parser("pipeline", help="Run api1 → api2 → api3 end-to-end")
    p4.add_argument("--topic", required=True)
    p4.add_argument("--goals", default="")
    p4.add_argument("--platforms", default="")
    p4.add_argument("--duration", default="")
    p4.add_argument("--audiences", default="")
    p4.add_argument("--presentation-mode", default="")
    p4.add_argument("--narration-mode", default="")
    p4.add_argument("--materials", default="")
    p4.add_argument("--extra-notes", default="")
    p4.add_argument(
        "--image",
        action="append",
        default=None,
        metavar="PATH",
        help="素材图片路径，可重复传入多张",
    )
    p4.add_argument(
        "--option-id",
        required=True,
        choices=("A", "B", "C"),
        help="Selected option id",
    )
    p4.add_argument(
        "--route",
        required=True,
        choices=("ai_instruction", "pro_script"),
    )
    p4.add_argument("--user-extra-request", default="")
    p4.add_argument(
        "--edited-fields",
        default="[]",
        help='JSON array, e.g. \'["core_decision.primary_goal"]\'',
    )
    p4.add_argument("--option-edited-fields", default="[]")
    p4.add_argument("--selected-option-data")
    p4.add_argument("--language", default="中文")
    p4.add_argument(
        "--out",
        help="Optional aggregate pipeline JSON output path; individual outputs are still written to outputs/api*/",
    )
    p4.set_defaults(func=_cmd_pipeline)

    args = parser.parse_args()
    try:
        asyncio.run(args.func(args))
    except (AIClientError, JSONParseError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()