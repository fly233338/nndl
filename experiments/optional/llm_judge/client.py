"""通过 OpenAI 兼容接口连接 vLLM 并校验描述评分。"""

import json
import os

from utils.config import load_yaml

SCORE_FIELDS = ("correctness", "coverage", "groundedness", "fluency")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "integer", "minimum": 1, "maximum": 5} for k in SCORE_FIELDS},
        "reason": {"type": "string"},
    },
    "required": [*SCORE_FIELDS, "reason"],
    "additionalProperties": False,
}


def connect(config):
    """连接服务并确认模型编号；仅在服务只有一个模型时自动选择。"""
    from openai import OpenAI

    api = config["api"]
    endpoint = os.environ.get(api["base_url_env"]) or api["base_url"]
    client = OpenAI(
        base_url=endpoint,
        api_key=os.environ.get(api["api_key_env"]) or "EMPTY",
        timeout=api["timeout"],
        max_retries=api["max_retries"],
    )
    try:
        available = [model.id for model in client.models.list().data]
        model = os.environ.get(api["model_env"]) or api["model"]
        if not model:
            if len(available) != 1:
                raise ValueError(f"Set JUDGE_MODEL to one of the served IDs: {available}")
            model = available[0]
        if model not in available:
            raise ValueError(
                f"Configured judge model {model!r} is not served; available IDs: {available}",
            )
        return client, model
    except Exception:
        client.close()
        raise


def request_scores(client, model, config, content):
    api = config["api"]
    args = {
        "model": model,
        "messages": [
            {"role": "system", "content": config["judge"]["system_prompt"]},
            {"role": "user", "content": content},
        ],
        "temperature": api["temperature"],
        "max_tokens": api["max_tokens"],
        "stream": False,
    }
    if api["structured_output"]:
        args["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "caption_judgment", "strict": True, "schema": JUDGE_SCHEMA},
        }
    if api.get("extra_body"):
        args["extra_body"] = api["extra_body"]
    return client.chat.completions.create(**args)


def parse_scores(raw):
    """严格校验四项整数评分和非空理由。"""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {*SCORE_FIELDS, "reason"}:
        raise ValueError("Judge must return exactly the four scores and reason")
    for field in SCORE_FIELDS:
        if type(value[field]) is not int or not 1 <= value[field] <= 5:
            raise ValueError(f"{field} must be an integer in 1..5")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Judge reason must be nonempty text")
    return value


def check_api(config_path):
    """检查接口和 JSON 格式，此请求不验证真实图片理解能力。"""
    config = load_yaml(config_path)
    client, model = connect(config)
    try:
        response = request_scores(
            client,
            model,
            config,
            "API format check only. No image is supplied. Return all four scores as 1 "
            "and explain that this is not an actual evaluation.",
        )
        result = parse_scores(response.choices[0].message.content or "")
        print(
            json.dumps(
                {
                    "model": model,
                    "chat_and_json": "ok",
                    "format_check": result,
                    "note": "Run --limit 1 on real inputs to check vision support.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    finally:
        client.close()
