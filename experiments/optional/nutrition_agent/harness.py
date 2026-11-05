"""执行查表、估重、可选复核和计算组成的有界工具循环。"""

import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO

from PIL import Image

from experiments.task2_meal_json.schema import canonical_json, parse_and_validate
from utils.files import sha256_file

from .food_names import FoodNameLookup
from .tools import NutritionDB, crop_image, normalize_weights, number

MODES = ("direct", "agent")


def declaration(name, description, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties) if required is None else required,
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
OBSERVATION = {
    "container": STRING,
    "servings": POSITIVE,
    "portion_level": {"type": "string", "enum": ["small", "medium", "large"]},
    "items": {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "name": STRING,
                "category": STRING,
                "ingredients": {"type": "array", "items": STRING},
                "cooking_method": STRING,
                "food_state": {"type": "string", "enum": ["raw", "cooked", "unknown"]},
                "sauce": {"type": "string", "enum": ["with", "without", "unknown"]},
                "mixed_dish": {"type": "boolean"},
                "servings": POSITIVE,
            },
            "required": [
                "name",
                "category",
                "ingredients",
                "cooking_method",
                "food_state",
                "sauce",
                "mixed_dish",
                "servings",
            ],
            "additionalProperties": False,
        },
    },
}
SELECTION = {
    "items": {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "food_id": STRING,
                "relative_weight": POSITIVE,
            },
            "required": ["food_id", "relative_weight"],
            "additionalProperties": False,
        },
    },
}
LOOKUP = declaration(
    "lookup_nutrition",
    "Observe the meal and look up each finished dish. "
    "On review, provide corrected observations for the whole meal.",
    OBSERVATION,
)
WEIGHT = declaration(
    "estimate_weight",
    "Predict whole-meal weight from the original RGB image using the independent frozen "
    "DINOv2 + Ridge model. No semantic inputs or arguments.",
    {},
)
CALCULATE = declaration(
    "calculate",
    "Select one food candidate per observed item IN ORDER; low-spread lookups use default_food_id. "
    "Allocate relative masses using the image and standard portions. "
    "Normalize to the independently predicted total and compute nutrition. Ends the loop.",
    SELECTION,
)
CROP = declaration(
    "crop_image",
    "Review one ambiguous region that could change food/state matching. Not a mass measurement.",
    {
        "bbox": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4,
            "maxItems": 4,
        },
        "reason": STRING,
    },
)


def calculation_tool(with_food_names=False):
    """仅在名称规范化实验中增加候选名称选择，原有工具声明保持不变。"""
    if not with_food_names:
        return CALCULATE
    tool = deepcopy(CALCULATE)
    item_schema = tool["function"]["parameters"]["properties"]["items"]["items"]
    item_schema["properties"]["canonical_name"] = {
        "type": ["string", "null"],
        "description": (
            "Choose a name from this item's food_name_candidates only if it names the same food. "
            "Use null when no candidate is equivalent; the original observed name is then kept."
        ),
    }
    item_schema["required"].append("canonical_name")
    return tool


def image_message(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        },
    }


def validate_observation(value):
    number(value["servings"], "servings", True)
    if (
        not isinstance(value["container"], str)
        or value["portion_level"] not in ("small", "medium", "large")
        or not value["items"]
    ):
        raise ValueError("Invalid container, portion_level or empty items")
    for item in value["items"]:
        for field in ("name", "category", "cooking_method"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Invalid {field}")
        if item["food_state"] not in ("raw", "cooked", "unknown") or item["sauce"] not in (
            "with",
            "without",
            "unknown",
        ):
            raise ValueError("Invalid food state or sauce")

        if (
            not isinstance(item["mixed_dish"], bool)
            or not isinstance(item["ingredients"], list)
            or any(not isinstance(x, str) for x in item["ingredients"])
        ):
            raise ValueError("Invalid ingredients or mixed_dish")
        number(item["servings"], "item servings", True)
    return value


def lookup_observation(db, observation, food_names=None):
    """保留营养候选，仅在启用时附加训练名称候选。"""
    lookups = []
    for item in observation["items"]:
        name_candidates = []
        if food_names is not None:
            name_candidates = food_names.lookup(item["name"], item["category"])
        try:
            lookup = db.lookup(
                item["name"],
                item["ingredients"],
                item["cooking_method"],
                item["food_state"],
                item["sauce"],
                item["mixed_dish"],
            )
        except ValueError as error:
            if not name_candidates:
                raise
            # 先返回名称候选，让模型使用原有复核机会纠正识别，不擅自查替代食品。
            lookup = {
                "candidates": [],
                "requires_choice": True,
                "default_food_id": None,
                "lookup_error": str(error),
            }
        if food_names is not None:
            lookup = {
                **lookup,
                "food_name_candidates": name_candidates,
            }
        lookups.append(lookup)
    return lookups


def select_items(observation, lookups, items):
    """按观察顺序选择食品和可选的规范名称，低密度差异项使用默认食品。"""
    if len(items) != len(observation["items"]):
        raise ValueError("Select exactly one candidate per observed item, preserving order")
    selected = []
    for item, lookup, selection in zip(observation["items"], lookups, items):
        name = item["name"]
        if "food_name_candidates" in lookup:
            canonical_name = selection.get("canonical_name")
            if canonical_name is not None:
                canonical_name = canonical_name.strip().lower()
                allowed_names = {entry["name"] for entry in lookup["food_name_candidates"]}
                if canonical_name not in allowed_names:
                    raise ValueError(
                        "Select canonical_name from this item's candidates or use null"
                    )
                name = canonical_name
        food_id = selection["food_id"] if lookup["requires_choice"] else lookup["default_food_id"]
        candidate = next((c for c in lookup["candidates"] if c["food_id"] == food_id), None)
        if candidate is None:
            raise ValueError(
                "Select a candidate returned for this item; review identity/state if needed",
            )
        selected.append(
            {
                "food_id": food_id,
                "name": name,
                "category": item["category"],
                "calibration_key": "|".join(
                    item[k].strip().lower()
                    for k in ("category", "cooking_method", "food_state", "sauce")
                ),
                "relative_weight": number(selection["relative_weight"], "relative_weight", True),
                "standard_portion_g": candidate["standard_portion_g"],
            },
        )
    return selected


def run_agent(image_path, config, mode="agent", observations_only=False):
    """仅将图片交给模型和工具，最多复核一次，最终数值由计算工具生成。"""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}")
    started = time.perf_counter()
    messages, calls = [], []
    requests = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw, client, observation = "", None, None
    reviews = 0
    image_hash = ""

    def finish(error=None):
        """隐藏图片编码并整理输出、调用轨迹和失败原因。"""
        trace_messages = []
        for message in messages:
            entry = dict(message)
            if isinstance(entry.get("content"), list):
                content = []
                for part in entry["content"]:
                    if part.get("type") == "image_url":
                        part = {
                            "type": "image_url",
                            "image_url": {"url": "omitted; see image hash and crop arguments"},
                        }
                    content.append(part)
                entry["content"] = content
            trace_messages.append(entry)

        if error is None and not observations_only:
            parsed = parse_and_validate(raw)
        else:
            parsed = {"valid": False, "parsed": None, "errors": [error] if error else []}
        return {
            "mode": mode,
            "status": "failed" if error else "success",
            "error": error,
            "raw_output": raw,
            "parsed": parsed,
            "model_requests": requests,
            "tool_call_count": len(calls),
            "review_count": reviews,
            "seconds": time.perf_counter() - started,
            "usage": usage,
            "trace": {"image_sha256": image_hash, "messages": trace_messages, "tool_calls": calls},
            **({"observation": observation} if observations_only and not error else {}),
        }

    try:
        image_hash = sha256_file(image_path)
        from openai import OpenAI

        api = config["api"]
        endpoint = os.environ.get(api["base_url_env"])
        if not endpoint:
            raise ValueError(f"Set {api['base_url_env']} to a vision/tool-calling Qwen endpoint")
        client = OpenAI(
            base_url=endpoint,
            api_key=os.environ.get(api["api_key_env"]) or "EMPTY",
            timeout=api["timeout"],
            max_retries=api["max_retries"],
        )
        prompt = (
            config["agent"]["system_prompt"]
            if mode != "direct"
            else (
                "Estimate the meal from the image. Return ONLY JSON: items [{name, weight_g}], "
                "total_weight_g, total_calories_kcal, protein_g, fat_g, carbohydrate_g. "
                "Numbers must be finite and nonnegative."
            )
        )
        food_names = None
        if mode == "agent" and config.get("food_names", {}).get("enabled", False):
            food_names = FoodNameLookup(config)
            prompt += (
                " The nutrition lookup also returns food_name_candidates from training labels. "
                "Use English observed names for lookup. In calculate, choose canonical_name "
                "only when a candidate means the same food at the observed granularity. "
                "Training frequency is not evidence that a food is visible. "
                "Do not replace a mixed dish by one ingredient or force an unseen food into "
                "the vocabulary. Use null when uncertain; preserve the original name. "
                "If lookup_error is returned, use the existing review to call lookup_nutrition "
                "with corrected whole-meal observations before calculate."
            )
        calculate = calculation_tool(food_names is not None)
        with Image.open(image_path) as image:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [image_message(image)]},
            ]
        db = None if mode == "direct" else NutritionDB(config, calibrated=not observations_only)
        stage = "lookup"
        lookups, weight = None, None
        # 先查表和估重，再决定是否复核；最多增加一次裁剪与重新查表。
        for _ in range(5):
            if mode == "direct":
                declarations = []
            elif stage == "lookup":
                declarations = [LOOKUP]
            elif stage == "weight":
                declarations = [WEIGHT]
            elif reviews < min(1, config["agent"]["max_reviews"]):
                declarations = [calculate, CROP, LOOKUP]
            else:
                declarations = [calculate]
            arguments = {
                "model": os.environ.get(api.get("model_env", "")) or api["model"],
                "messages": messages,
                "temperature": api["temperature"],
                "max_tokens": api["max_tokens"],
                "stream": False,
            }
            if declarations:
                choice = (
                    {"type": "function", "function": {"name": declarations[0]["function"]["name"]}}
                    if len(declarations) == 1
                    else "required"
                )
                arguments.update(tools=declarations, tool_choice=choice, parallel_tool_calls=False)
            if api.get("extra_body"):
                arguments["extra_body"] = api["extra_body"]
            requests += 1
            response = client.chat.completions.create(**arguments)
            message = response.choices[0].message
            messages.append(
                {
                    k: v
                    for k, v in message.model_dump(exclude_none=True).items()
                    if k in ("role", "content", "tool_calls", "reasoning_content")
                },
            )
            for key, value in usage.items():
                if response.usage and value is not None:
                    usage[key] = value + (getattr(response.usage, key, 0) or 0)
                else:
                    usage[key] = None
            if mode == "direct":
                raw = message.content or ""
                return finish(None if parse_and_validate(raw)["valid"] else "invalid_final_json")
            tool_calls = message.tool_calls or []
            if len(tool_calls) != 1:
                raise ValueError("Exactly one tool call is required per stage")
            call = tool_calls[0]
            name = call.function.name
            if name not in {d["function"]["name"] for d in declarations}:
                raise ValueError(f"Tool {name} is unavailable in stage {stage}")
            values = json.loads(call.function.arguments)
            crop = None
            if name == "lookup_nutrition":
                if stage == "decide":
                    reviews += 1
                observation = validate_observation(values)
                lookups = lookup_observation(db, observation, food_names)
                result = {"items": lookups}
                stage = "weight" if weight is None else "decide"
            elif name == "estimate_weight":
                if values != {}:
                    raise ValueError(
                        "estimate_weight accepts no semantic arguments; "
                        "it uses only the original RGB image",
                    )
                from .weight import estimate_weight

                weight = estimate_weight(image_path, config)
                result = weight
                stage = "decide"
            elif name == "crop_image":
                reviews += 1
                crop = crop_image(image_path, values["bbox"], config["agent"]["crop_max_size"])
                result = {
                    "review": reviews,
                    "reason": values["reason"],
                    "next": (
                        "Inspect the crop, then call lookup_nutrition with corrected "
                        "whole-meal observations. No further review allowed."
                    ),
                }
                stage = "lookup"
            else:
                final_items = select_items(observation, lookups, values["items"])
                result = db.calculate(normalize_weights(final_items, weight["total_weight_g"]))
                raw = canonical_json(result)
            calls.append({"name": name, "arguments": values, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, allow_nan=False),
                },
            )
            if observations_only and name == "lookup_nutrition":
                return finish()

            if crop is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Cropped detail of the original image; "
                                    "reassess identity/cooking only."
                                ),
                            },
                            image_message(crop),
                        ],
                    },
                )
            if name == "calculate":
                if not parse_and_validate(raw)["valid"]:
                    raise ValueError("Calculated JSON failed schema validation")
                return finish()
        return finish("request_budget_exhausted")
    # 逐图错误需要保留，批量推理才能继续并统计失败率。
    except Exception as exc:  # noqa: BLE001
        return finish(f"{type(exc).__name__}: {exc}")
    finally:
        if client is not None:
            client.close()
