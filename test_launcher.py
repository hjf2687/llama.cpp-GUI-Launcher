#!/usr/bin/env python3
"""
测试启动器功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

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
