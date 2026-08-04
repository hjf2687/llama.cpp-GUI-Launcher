#!/usr/bin/env python3
"""
模型预设管理器
================
每个模型拥有独立的采样参数预设集合，切换模型时自动加载该模型的当前预设。

数据存储: model_presets.json（与 launcher_config.json 同目录）
结构:
{
  "models": {
    "<模型文件路径>": {
      "display_name": "模型显示名",
      "current_preset": "当前预设名",
      "presets": {
        "预设名": {
          "temperature": 1.0,
          "top_p": 0.95,
          "top_k": 20,
          "min_p": 0.0,
          "repeat_penalty": 1.1,
          "frequency_penalty": 0.0,
          "presence_penalty": 1.5
        },
        ...
      }
    },
    ...
  }
}
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# 预设包含的采样参数键（与 llama_launcher.py 的采样参数 UI 保持一致）
SAMPLING_KEYS = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "frequency_penalty",
    "presence_penalty",
]

# 采样参数默认值（与 llama_launcher.py 的 load_config 默认值一致）
DEFAULT_SAMPLING: Dict[str, float] = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# ======================== 推荐预设 ========================
# 根据模型名匹配（35B / 9B / 4B），首次选择该模型且无预设时自动创建。
# 数值来自用户提供的官方推荐配置文档。

_RECOMMENDED_35B = {
    "思考-概述": {**DEFAULT_SAMPLING, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "思考-编码": {**DEFAULT_SAMPLING, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "非思考-概述": {**DEFAULT_SAMPLING, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "非思考-推理": {**DEFAULT_SAMPLING, "temperature": 1.0, "top_p": 1.0, "top_k": 40, "min_p": 0.0, "presence_penalty": 2.0},
}

_RECOMMENDED_SMALL = {
    "思考模式": {**DEFAULT_SAMPLING, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "非思考模式": {**DEFAULT_SAMPLING, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
}

# 匹配顺序: 35B 优先（避免 "35B" 与 "9B"/"4B" 混淆）
RECOMMENDED_PRESETS: Dict[str, Dict[str, Dict]] = {
    "35B": _RECOMMENDED_35B,
    "9B": _RECOMMENDED_SMALL,
    "4B": _RECOMMENDED_SMALL,
}

# 匹配 "35B" / "9B" / "4B" 之类的独立 token（整体匹配，避免 GLM-4-9B 误判为 4B）
_RECOMMEND_RE = re.compile(r'(?<![0-9A-Za-z.])(35B|9B|4B)(?![0-9A-Za-z])', re.IGNORECASE)


def _match_recommend_token(display_name: str) -> Optional[str]:
    """返回匹配到的推荐档位 token（"35B"/"9B"/"4B"），未匹配返回 None"""
    if not display_name:
        return None
    m = _RECOMMEND_RE.search(display_name)
    if not m:
        return None
    return m.group(0).strip().upper()


def match_recommended_presets(display_name: str) -> Optional[Dict[str, Dict]]:
    """根据模型显示名匹配推荐预设表，未匹配返回 None"""
    token = _match_recommend_token(display_name)
    if not token:
        return None
    return RECOMMENDED_PRESETS.get(token)


class PresetManager:
    """每个模型的采样参数预设集合管理"""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path(__file__).parent / "model_presets.json"
        self.data = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("models"), dict):
                    return data
            except Exception as e:
                print(f"加载预设文件失败: {e}")
        return {"models": {}}

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存预设文件失败: {e}")

    # ---------------- 模型条目 ----------------

    def get_model_entry(self, model_key: str, display_name: str = "") -> Dict:
        """获取模型条目（不存在时在内存中创建，不写盘）"""
        models = self.data["models"]
        if model_key not in models:
            models[model_key] = {
                "display_name": display_name,
                "current_preset": "",
                "presets": {},
            }
        elif display_name and models[model_key].get("display_name") != display_name:
            models[model_key]["display_name"] = display_name
        return models[model_key]

    def has_model_entry(self, model_key: str) -> bool:
        """该模型是否已有预设记录（无论预设是否为空）"""
        return model_key in self.data["models"]

    def get_preset_names(self, model_key: str) -> List[str]:
        entry = self.data["models"].get(model_key)
        if not entry:
            return []
        return list(entry.get("presets", {}).keys())

    def get_current_preset_name(self, model_key: str) -> str:
        entry = self.data["models"].get(model_key)
        if not entry:
            return ""
        name = entry.get("current_preset", "")
        if name not in entry.get("presets", {}):
            return ""
        return name

    def get_preset_params(self, model_key: str, preset_name: str) -> Optional[Dict]:
        entry = self.data["models"].get(model_key)
        if not entry:
            return None
        params = entry.get("presets", {}).get(preset_name)
        if not params:
            return None
        # 补齐缺失键，保证 UI 应用时字段完整
        merged = dict(DEFAULT_SAMPLING)
        merged.update({k: params[k] for k in SAMPLING_KEYS if k in params})
        return merged

    # ---------------- 预设操作 ----------------

    def save_preset(self, model_key: str, preset_name: str, params: Dict,
                    display_name: str = "") -> str:
        """保存（或覆盖）一个预设，并设为当前预设"""
        name = preset_name.strip()
        if not name:
            raise ValueError("预设名称不能为空")
        entry = self.get_model_entry(model_key, display_name)
        entry["presets"][name] = {
            k: params.get(k, DEFAULT_SAMPLING[k]) for k in SAMPLING_KEYS
        }
        entry["current_preset"] = name
        self._save()
        return name

    def delete_preset(self, model_key: str, preset_name: str) -> bool:
        """删除指定预设；若删除的是当前预设，自动切换到剩余第一个"""
        entry = self.data["models"].get(model_key)
        if not entry:
            return False
        presets = entry.get("presets", {})
        if preset_name not in presets:
            return False
        del presets[preset_name]
        if entry.get("current_preset") == preset_name:
            entry["current_preset"] = next(iter(presets), "")
        self._save()
        return True

    def set_current_preset(self, model_key: str, preset_name: str,
                           display_name: str = "") -> bool:
        """把某个已存在的预设设为当前预设；无变化时不写盘"""
        entry = self.get_model_entry(model_key, display_name)
        if preset_name not in entry.get("presets", {}):
            return False
        if entry.get("current_preset") == preset_name:
            return True
        entry["current_preset"] = preset_name
        self._save()
        return True

    def create_recommended_presets(self, model_key: str, display_name: str) -> bool:
        """
        按模型名自动创建推荐预设。
        仅在该模型从未有过预设记录（磁盘/内存中都不存在该条目）时创建，
        避免用户删除全部预设后推荐预设被静默重建。创建成功返回 True。
        """
        if model_key in self.data["models"]:
            return False
        recommended = match_recommended_presets(display_name)
        if not recommended:
            return False
        entry = self.get_model_entry(model_key, display_name)
        for name, params in recommended.items():
            entry["presets"][name] = dict(params)
        entry["current_preset"] = next(iter(recommended))
        self._save()
        return True

    def remove_model(self, model_key: str) -> bool:
        """删除某模型的所有预设"""
        if model_key in self.data["models"]:
            del self.data["models"][model_key]
            self._save()
            return True
        return False
