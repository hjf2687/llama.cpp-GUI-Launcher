#!/usr/bin/env python3
"""
测试启动器功能
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from preset_manager import (
    PresetManager,
    SAMPLING_KEYS,
    match_recommended_presets,
    _match_recommend_token,
)

def test_model_scanning():
    """测试模型扫描功能"""
    print("测试模型扫描...")

    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        print("[FAIL] models目录不存在")
        return False

    # 扫描模型
    models = []
    for model_dir in models_dir.rglob("*.gguf"):
        if "mmproj" in model_dir.name.lower():
            continue

        # 查找mmproj文件
        mmproj_file = None
        for f in model_dir.parent.glob("mmproj-*.gguf"):
            if model_dir.stem in f.stem:
                mmproj_file = f
                break

        models.append({
            "name": model_dir.stem,
            "path": str(model_dir),
            "has_multimodal": mmproj_file is not None
        })

    if models:
        print(f"[OK] 找到 {len(models)} 个模型:")
        for model in models:
            multimodal = "[多模态]" if model["has_multimodal"] else "[无多模态]"
            print(f"  - {model['name']} {multimodal}")
        return True
    else:
        print("[FAIL] 未找到任何模型")
        return False


def test_executables():
    """测试可执行文件"""
    print("\n测试可执行文件...")

    exe_dir = Path(__file__).parent
    required_exes = [
        "llama-server.exe",
        "llama-cli.exe",
        "llama-completion.exe",
        "llama-bench.exe"
    ]

    all_exist = True
    for exe in required_exes:
        exe_path = exe_dir / exe
        if exe_path.exists():
            print(f"[OK] {exe}")
        else:
            print(f"[FAIL] {exe} 不存在")
            all_exist = False

    return all_exist


def test_config():
    """测试配置管理"""
    print("\n测试配置管理...")

    config_file = Path(__file__).parent / "launcher_config.json"

    # 测试配置加载
    try:
        import json
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[OK] 配置文件加载成功: {len(config)} 个配置项")
        else:
            print("[WARN] 配置文件不存在，将使用默认配置")
        return True
    except Exception as e:
        print(f"[FAIL] 配置加载失败: {e}")
        return False


def test_command_generation():
    """测试命令生成"""
    print("\n测试命令生成...")

    # 模拟配置
    config = {
        "gpu_layers": "auto",
        "threads": -1,
        "ctx_size": 4096,
        "batch_size": 2048,
        "flash_attention": "auto",
        "kv_cache_type": "f16",
        "mlock": False,
        "mmap": True,
        "embedding": False,
        "host": "127.0.0.1",
        "port": 8080
    }

    # 模拟模型信息
    model = {
        "path": "D:/tools/AI/llama/models/qwen3.6 9B/qwen3.6 9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "has_multimodal": True,
        "mmproj_path": "D:/tools/AI/llama/models/qwen3.6 9B/qwen3.6 9B/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    }

    # 测试服务器模式
    args = ["-m", model["path"]]
    args.extend(["-ngl", str(config["gpu_layers"])])
    args.extend(["-c", str(config["ctx_size"])])
    args.extend(["-b", str(config["batch_size"])])
    args.extend(["--host", config["host"]])
    args.extend(["--port", str(config["port"])])

    if config["mmap"]:
        pass  # 默认启用
    else:
        args.append("--no-mmap")

    print(f"[OK] 服务器模式命令参数: {len(args)} 个参数")
    print(f"  示例: llama-server.exe {' '.join(args[:5])}...")

    return True


def test_preset_manager():
    """测试每个模型的独立预设管理"""
    print("\n测试模型预设管理...")

    model_a = "D:/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_b = "D:/models/Qwen3.5-9B-Q4_K_M.gguf"

    with tempfile.TemporaryDirectory() as tmp:
        pm_path = Path(tmp) / "model_presets.json"
        pm = PresetManager(str(pm_path))

        # 1. 保存预设（模型A 两个预设，模型B 一个预设）
        pm.save_preset(model_a, "思考-概述",
                       {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 1.5},
                       "Qwen3.6-35B")
        pm.save_preset(model_a, "思考-编码",
                       {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
                       "Qwen3.6-35B")
        pm.save_preset(model_b, "思考模式",
                       {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
                       "Qwen3.5-9B")

        # 2. 两个模型的预设互相独立
        assert pm.get_preset_names(model_a) == ["思考-概述", "思考-编码"], pm.get_preset_names(model_a)
        assert pm.get_preset_names(model_b) == ["思考模式"], pm.get_preset_names(model_b)

        # 3. 保存后自动设为当前预设
        assert pm.get_current_preset_name(model_a) == "思考-编码"
        assert pm.get_current_preset_name(model_b) == "思考模式"

        # 4. 读取预设参数
        params = pm.get_preset_params(model_a, "思考-概述")
        assert params["temperature"] == 1.0 and params["presence_penalty"] == 1.5
        assert set(SAMPLING_KEYS) <= set(params.keys())  # 键完整

        # 5. 覆盖同名预设
        pm.save_preset(model_a, "思考-概述",
                       {"temperature": 0.9, "top_p": 0.9, "top_k": 30, "min_p": 0.0,
                        "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 1.0},
                       "Qwen3.6-35B")
        assert pm.get_preset_params(model_a, "思考-概述")["temperature"] == 0.9

        # 6. 切换当前预设
        assert pm.set_current_preset(model_a, "思考-概述") is True
        assert pm.get_current_preset_name(model_a) == "思考-概述"
        assert pm.set_current_preset(model_a, "不存在的预设") is False

        # 7. 删除非当前预设；再删除当前预设后自动切到剩余第一个
        pm.save_preset(model_b, "非思考模式",
                       {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                        "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
                       "Qwen3.5-9B")
        assert pm.delete_preset(model_b, "思考模式") is True
        assert pm.get_current_preset_name(model_b) == "非思考模式"

        # 8. 持久化：重新加载文件
        pm2 = PresetManager(str(pm_path))
        assert pm2.get_preset_names(model_a) == ["思考-概述", "思考-编码"]
        assert pm2.get_current_preset_name(model_a) == "思考-概述"
        assert pm2.get_preset_params(model_a, "思考-编码")["temperature"] == 0.6

        # 9. 推荐预设匹配
        assert match_recommended_presets("Qwen3.6-35B-A3B-Q4_K_M") is not None
        assert match_recommended_presets("Qwen3.5-9B-Q4_K_M") is not None
        assert match_recommended_presets("Qwen3.5-4B-Q4_K_M") is not None
        assert match_recommended_presets("Qwen2.5-7B-Q4_K_M") is None
        assert match_recommended_presets("Qwen0.4B") is None  # 0.4B 不应误判为 4B
        # token 精确匹配（GLM-4-9B 应命中 9B 而非 4B）
        assert _match_recommend_token("Qwen3.6-35B-A3B-Q4_K_M") == "35B"
        assert _match_recommend_token("Qwen3.5-9B-Q4_K_M") == "9B"
        assert _match_recommend_token("Qwen3.5-4B-Q4_K_M") == "4B"
        assert _match_recommend_token("GLM-4-9B-Q4_K_M") == "9B"
        assert _match_recommend_token("Qwen0.4B") is None

        # 10. 删除全部预设后不会自动重建推荐预设（避免"复活"）
        pm.delete_preset(model_b, "非思考模式")
        assert pm.get_preset_names(model_b) == []
        assert pm.create_recommended_presets(model_b, "Qwen3.5-9B-Q4_K_M") is False
        # 从未出现过的新模型才会创建推荐预设
        model_c = "D:/models/Qwen3.5-9B-Other.gguf"
        assert pm.create_recommended_presets(model_c, "Qwen3.5-9B-Q4_K_M") is True
        assert "思考模式" in pm.get_preset_names(model_c)
        # 已有预设时不重复创建
        assert pm.create_recommended_presets(model_c, "Qwen3.5-9B-Q4_K_M") is False

    print("[OK] 模型预设管理功能正常")
    return True


def main():
    """主测试函数"""
    print("=" * 50)
    print("llama.cpp 启动器功能测试")
    print("=" * 50)

    results = []

    # 运行测试
    results.append(("模型扫描", test_model_scanning()))
    results.append(("可执行文件", test_executables()))
    results.append(("配置管理", test_config()))
    results.append(("命令生成", test_command_generation()))
    results.append(("模型预设管理", test_preset_manager()))

    # 显示结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    passed = 0
    failed = 0

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("\n[SUCCESS] 所有测试通过！启动器可以正常使用。")
        return True
    else:
        print("\n[WARNING] 部分测试失败，请检查上述问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
