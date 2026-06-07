"""配置加载模块。

读取根目录 config.yaml，提供类型安全的配置访问。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _repo_root() -> Path:
    """返回项目根目录（config.yaml 所在目录）。"""
    return Path(__file__).resolve().parent.parent


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        path: 配置文件路径；默认使用项目根目录的 config.yaml。

    Returns:
        配置字典。
    """
    if path is None:
        path = str(_repo_root() / "config.yaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析相对路径
    cfg["_repo_root"] = _repo_root()
    return cfg


def resolve_path(raw: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """将配置中的相对路径解析为绝对路径。"""
    p = Path(raw)
    if p.is_absolute():
        return p
    if config and "_repo_root" in config:
        return config["_repo_root"] / raw
    return _repo_root() / raw
