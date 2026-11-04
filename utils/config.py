"""读取 YAML 配置，并递归合并相对路径指定的基础配置。"""

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """相对当前配置解析 base，并递归覆盖嵌套字典。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base_path = config.pop("base", None)
    if base_path:
        base = load_yaml(path.parent / base_path)
        config = _merge(base, config)
    return config
