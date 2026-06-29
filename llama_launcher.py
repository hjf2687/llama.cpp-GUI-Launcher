#!/usr/bin/env python3
"""
llama.cpp GUI Launcher - Optimized Version
图形界面启动器，用于管理llama.cpp程序
优化内容：
  - 修复 Flash Attention UI 重叠 bug
  - 修复 save_config multimodal_enabled 默认值不一致
  - 将 import re 移到文件顶部
  - 新增 GPU 自动检测
  - 新增端口占用检测
  - 新增一键打开 Web UI 按钮
  - 新增日志颜色高亮（错误/警告行红色/橙色显示）
  - 新增深色模式支持
  - 新增启动前检查（模型是否存在、端口是否被占用等）
  - 优化 UI 布局，使用 Notebook 分页减少拥挤感
"""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import signal
import time
import re


class LlamaLauncher:
    """llama.cpp启动器主类"""

    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp 启动器")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # 设置图标（如果存在）
        icon_path = Path(__file__).parent / "llama.ico"
        if icon_path.exists():
            self.root.iconbitmap(str(icon_path))

        # 当前运行的进程
        self.process = None
        self.process_thread = None
        self.monitor_thread = None
        self.monitor_running = False

        # 主题状态
        self.dark_mode = False

        # 配置文件路径
        self.config_file = Path(__file__).parent / "launcher_config.json"
        self.config = self.load_config()

        # GPU 信息缓存
        self.gpu_info = self.detect_gpu()

        # 扫描模型
        self.models = self.scan_models()

        # 创建GUI
        self.create_widgets()

        # 加载上次配置
        self.load_last_config()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ======================== 配置管理 ========================

    def load_config(self) -> Dict:
        """加载配置文件"""
        default_config = {
            "last_model": "",
            "last_mode": "server",
            "gpu_layers": "auto",
            "threads": -1,
            "ctx_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "host": "127.0.0.1",
            "port": 8080,
            "flash_attention": "auto",
            "multimodal_enabled": False,
            "kv_cache_type": "f16",
            "mlock": False,
            "mmap": True,
            "embedding": False,
            "dark_mode": False,
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    default_config.update(saved_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")

        return default_config

    def save_config(self):
        """保存配置文件"""
        try:
            self.config["last_model"] = self.model_var.get()
            self.config["last_mode"] = self.mode_var.get()
            self.config["gpu_layers"] = self.ngl_var.get()
            self.config["threads"] = self.threads_var.get()
            self.config["ctx_size"] = self.ctx_size_var.get()
            self.config["batch_size"] = self.batch_size_var.get()
            self.config["ubatch_size"] = self.ubatch_size_var.get()
            self.config["host"] = self.host_var.get()
            self.config["port"] = self.port_var.get()
            self.config["flash_attention"] = self.flash_attn_var.get()
            self.config["multimodal_enabled"] = self.multimodal_var.get()
            self.config["kv_cache_type"] = self.kv_cache_var.get()
            self.config["mlock"] = self.mlock_var.get()
            self.config["mmap"] = self.mmap_var.get()
            self.config["embedding"] = self.embedding_var.get()
            self.config["tools"] = self.tools_var.get()
            self.config["dark_mode"] = self.dark_mode

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")

    # ======================== GPU 检测 ========================

    def detect_gpu(self) -> Dict:
        """检测系统中可用的 GPU"""
        gpu_info = {"cuda": False, "cuda_version": "", "gpu_name": "", "vram_mb": 0}

        # 检测 NVIDIA GPU (via nvidia-smi)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(", ")
                    if len(parts) >= 2:
                        gpu_info["cuda"] = True
                        gpu_info["gpu_name"] = parts[0]
                        gpu_info["vram_mb"] = int(parts[1])
        except Exception:
            pass

        # 检测 CUDA 可用 (via cublas DLL version)
        cublas_path = Path(__file__).parent / "cublas64_13.dll"
        if cublas_path.exists():
            gpu_info["cuda_dll"] = True

        return gpu_info

    # ======================== 模型扫描 ========================

    def scan_models(self) -> List[Dict]:
        """扫描模型目录，识别可用模型"""
        models = []
        models_dir = Path(__file__).parent / "models"

        if not models_dir.exists():
            return models

        for model_dir in models_dir.rglob("*.gguf"):
            # 排除mmproj文件
            if "mmproj" in model_dir.name.lower():
                continue

            # 查找对应的mmproj文件
            mmproj_file = None

            # 提取模型基础名称（去掉量化类型后缀）
            base_name = re.sub(
                r'[-_](Q[2-8]_[A-Z0-9_]+|IQ[0-9]+_[A-Z0-9_]+|F16|BF16|FP16|FP32|Q8_0)$',
                '', model_dir.stem, flags=re.IGNORECASE
            )

            mmproj_patterns = [
                f"mmproj-{model_dir.stem}.gguf",
                f"mmproj-{base_name}.gguf",
            ]

            for pattern in mmproj_patterns:
                candidate = model_dir.parent / pattern
                if candidate.exists():
                    mmproj_file = candidate
                    break

            if mmproj_file is None:
                for f in model_dir.parent.glob("mmproj-*.gguf"):
                    if base_name in f.stem:
                        mmproj_file = f
                        break
                    elif f.stem.replace("mmproj-", "").replace("-mmproj", "") in model_dir.stem:
                        mmproj_file = f
                        break

            model_info = {
                "name": model_dir.stem,
                "path": str(model_dir),
                "size_mb": model_dir.stat().st_size / (1024 * 1024),
                "has_multimodal": mmproj_file is not None,
                "mmproj_path": str(mmproj_file) if mmproj_file else None
            }

            parent_name = model_dir.parent.name
            if parent_name and parent_name != model_dir.stem:
                model_info["display_name"] = f"{parent_name} - {model_dir.stem}"
            else:
                model_info["display_name"] = model_dir.stem

            models.append(model_info)

        models.sort(key=lambda x: x["size_mb"], reverse=True)
        return models

    # ======================== GUI 创建 ========================

    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 左侧面板 - 配置区 (使用 Notebook 分页)
        left_outer = ttk.Frame(main_frame)
        left_outer.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(left_outer)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 分页1: 基本配置
        config_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(config_tab, text="基本配置")
        self.create_basic_config_widgets(config_tab)

        # 分页2: 高级参数
        advanced_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(advanced_tab, text="高级参数")
        self.create_advanced_config_widgets(advanced_tab)

        # 左侧底部按钮区
        self.create_button_area(left_outer)

        # 右侧面板 - 输出区
        right_frame = ttk.LabelFrame(main_frame, text="输出日志", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.create_output_widgets(right_frame)

        # 应用主题
        if self.config.get("dark_mode", False):
            self.toggle_dark_mode(apply_only=True)

    def create_basic_config_widgets(self, parent):
        """创建基本配置组件"""
        parent.columnconfigure(0, weight=1)

        # 模型选择
        model_frame = ttk.LabelFrame(parent, text="模型选择", padding="8")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        model_frame.columnconfigure(0, weight=1)

        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[m["display_name"] for m in self.models],
            state="readonly",
            width=40
        )
        self.model_combo.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        self.model_info_label = ttk.Label(model_frame, text="", wraplength=350)
        self.model_info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # GPU 信息显示
        gpu_text = "GPU: "
        if self.gpu_info.get("cuda"):
            gpu_text += f"{self.gpu_info['gpu_name']} ({self.gpu_info['vram_mb']}MB)"
        elif self.gpu_info.get("cuda_dll"):
            gpu_text += "CUDA 可用 (未检测到 NVIDIA GPU)"
        else:
            gpu_text += "未检测到 GPU，将使用 CPU"
        ttk.Label(model_frame, text=gpu_text, foreground="blue").grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0)
        )

        # 启动模式
        mode_frame = ttk.LabelFrame(parent, text="启动模式", padding="8")
        mode_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.mode_var = tk.StringVar(value=self.config["last_mode"])

        modes = [
            ("API 服务器", "server"),
            ("CLI 交互", "cli"),
            ("文本补全", "completion"),
            ("基准测试", "bench")
        ]

        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self.on_mode_change
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=(0, 15), pady=2)

        # GPU 配置
        gpu_frame = ttk.LabelFrame(parent, text="GPU 配置", padding="8")
        gpu_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        ttk.Label(gpu_frame, text="GPU 层数 (-ngl):").grid(row=0, column=0, sticky=tk.W)

        self.ngl_var = tk.StringVar(value=str(self.config["gpu_layers"]))
        ngl_values = ["auto", "all"] + [str(i) for i in [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 99]]
        ngl_combo = ttk.Combobox(
            gpu_frame,
            textvariable=self.ngl_var,
            values=ngl_values,
            width=12
        )
        ngl_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        ttk.Label(gpu_frame, text="(auto=自动, all=全部加载到GPU)").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0)
        )

        # 服务器配置
        self.server_frame = ttk.LabelFrame(parent, text="服务器配置", padding="8")
        self.server_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        ttk.Label(self.server_frame, text="监听地址:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.host_var = tk.StringVar(value=self.config["host"])
        host_entry = ttk.Entry(self.server_frame, textvariable=self.host_var, width=18)
        host_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.server_frame, text="端口:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.port_var = tk.IntVar(value=self.config["port"])
        port_spin = ttk.Spinbox(
            self.server_frame,
            from_=1024,
            to=65535,
            textvariable=self.port_var,
            width=12
        )
        port_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 端口检测标签
        self.port_status_label = ttk.Label(self.server_frame, text="", foreground="gray")
        self.port_status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        # 一键打开 Web UI 按钮 (仅 server 模式)
        self.webui_button = ttk.Button(
            self.server_frame,
            text="🌐 打开 Web UI",
            command=self.open_webui,
            state=tk.DISABLED
        )
        self.webui_button.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # 端口检测放到后台，避免界面卡住
        self.root.after(100, self.check_port_status)

    def create_advanced_config_widgets(self, parent):
        """创建高级参数配置组件"""
        parent.columnconfigure(0, weight=1)

        # 参数配置
        params_frame = ttk.LabelFrame(parent, text="性能参数", padding="8")
        params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # 线程数
        ttk.Label(params_frame, text="线程数 (-t):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threads_var = tk.IntVar(value=self.config["threads"])
        threads_spin = ttk.Spinbox(
            params_frame,
            from_=-1,
            to=64,
            textvariable=self.threads_var,
            width=12
        )
        threads_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(params_frame, text="(-1=自动)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # 上下文大小
        ttk.Label(params_frame, text="上下文大小 (-c):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ctx_size_var = tk.IntVar(value=self.config["ctx_size"])
        ctx_spin = ttk.Spinbox(
            params_frame,
            from_=512,
            to=131072,
            increment=1024,
            textvariable=self.ctx_size_var,
            width=12
        )
        ctx_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 批处理大小
        ttk.Label(params_frame, text="批处理大小 (-b):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=self.config["batch_size"])
        batch_spin = ttk.Spinbox(
            params_frame,
            from_=64,
            to=8192,
            increment=64,
            textvariable=self.batch_size_var,
            width=12
        )
        batch_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 微批处理大小
        ttk.Label(params_frame, text="微批处理大小 (-ub):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.ubatch_size_var = tk.IntVar(value=self.config["ubatch_size"])
        ubatch_spin = ttk.Spinbox(
            params_frame,
            from_=16,
            to=4096,
            increment=16,
            textvariable=self.ubatch_size_var,
            width=12
        )
        ubatch_spin.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Flash Attention (FIX: row=4 与 Label 对齐)
        ttk.Label(params_frame, text="Flash Attention (-fa):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.flash_attn_var = tk.StringVar(value=self.config["flash_attention"])
        flash_combo = ttk.Combobox(
            params_frame,
            textvariable=self.flash_attn_var,
            values=["auto", "on", "off"],
            state="readonly",
            width=12
        )
        flash_combo.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # KV Cache 类型
        ttk.Label(params_frame, text="KV Cache 类型:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.kv_cache_var = tk.StringVar(value=self.config["kv_cache_type"])
        kv_combo = ttk.Combobox(
            params_frame,
            textvariable=self.kv_cache_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            state="readonly",
            width=12
        )
        kv_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 多模态支持
        multimodal_frame = ttk.LabelFrame(parent, text="多模态支持", padding="8")
        multimodal_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.multimodal_var = tk.BooleanVar(value=self.config["multimodal_enabled"])
        self.multimodal_check = ttk.Checkbutton(
            multimodal_frame,
            text="启用多模态（图像理解）",
            variable=self.multimodal_var,
            command=self.on_multimodal_change
        )
        self.multimodal_check.grid(row=0, column=0, sticky=tk.W)

        self.multimodal_info = ttk.Label(
            multimodal_frame,
            text="",
            wraplength=350,
            foreground="gray"
        )
        self.multimodal_info.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

        # 高级选项
        advanced_frame = ttk.LabelFrame(parent, text="高级选项", padding="8")
        advanced_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.mlock_var = tk.BooleanVar(value=self.config["mlock"])
        ttk.Checkbutton(
            advanced_frame,
            text="锁定内存 (--mlock)",
            variable=self.mlock_var
        ).grid(row=0, column=0, sticky=tk.W, pady=2)

        self.mmap_var = tk.BooleanVar(value=self.config["mmap"])
        ttk.Checkbutton(
            advanced_frame,
            text="内存映射 (mmap, 取消则 --no-mmap)",
            variable=self.mmap_var
        ).grid(row=1, column=0, sticky=tk.W, pady=2)

        self.embedding_var = tk.BooleanVar(value=self.config["embedding"])
        ttk.Checkbutton(
            advanced_frame,
            text="嵌入模式 (--embedding)",
            variable=self.embedding_var
        ).grid(row=2, column=0, sticky=tk.W, pady=2)

        self.tools_var = tk.BooleanVar(value=self.config.get("tools", False))
        ttk.Checkbutton(
            advanced_frame,
            text="内置工具 (--tools, 启用 function calling)",
            variable=self.tools_var
        ).grid(row=3, column=0, sticky=tk.W, pady=2)

    def create_button_area(self, parent):
        """创建按钮区"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # 启动按钮
        self.start_button = ttk.Button(
            button_frame,
            text="▶ 启动",
            command=self.start_process,
            style="Accent.TButton" if "Accent.TButton" in ttk.Style().theme_names() else ""
        )
        if self.start_button.cget("style") == "":
            self.start_button.config(style="")
        self.start_button.grid(row=0, column=0, padx=(0, 5), pady=2)

        self.stop_button = ttk.Button(
            button_frame,
            text="⏹ 停止",
            command=self.stop_process,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 5), pady=2)

        # 保存配置按钮
        save_btn = ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.save_config
        )
        save_btn.grid(row=0, column=2, padx=(0, 5), pady=2)

        # 刷新模型按钮
        refresh_btn = ttk.Button(
            button_frame,
            text="🔄 刷新模型",
            command=self.refresh_models
        )
        refresh_btn.grid(row=0, column=3, padx=(0, 5), pady=2)

        # 深色模式切换
        self.dark_btn = ttk.Button(
            button_frame,
            text="🌙 深色模式" if not self.dark_mode else "☀️ 浅色模式",
            command=self.toggle_dark_mode
        )
        self.dark_btn.grid(row=0, column=4, padx=(0, 5), pady=2)

    def create_output_widgets(self, parent):
        """创建输出区组件"""
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # 日志文本框 (等宽字体，支持颜色)
        self.log_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            width=65,
            height=28,
            font=("Consolas", 10 if not self.dark_mode else 10),
            bg="#ffffff" if not self.dark_mode else "#1e1e1e",
            fg="#000000" if not self.dark_mode else "#d4d4d4",
            insertbackground="#000000" if not self.dark_mode else "#ffffff"
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 定义颜色标签
        self.log_text.tag_config("error", foreground="#ff0000")
        self.log_text.tag_config("warning", foreground="#ff8c00")
        self.log_text.tag_config("info", foreground="#0000ff")
        self.log_text.tag_config("success", foreground="#008000")

        # 底部按钮栏
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(bottom_frame, text="清除日志", command=self.clear_log).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(bottom_frame, text="复制日志", command=self.copy_log).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            bottom_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_label.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(10, 0))
        bottom_frame.columnconfigure(2, weight=1)

    # ======================== 事件处理 ========================

    def load_last_config(self):
        """加载上次的配置"""
        if self.models:
            last_model = self.config.get("last_model", "")
            if last_model:
                for i, model in enumerate(self.models):
                    if model["display_name"] == last_model or model["name"] == last_model:
                        self.model_combo.current(i)
                        break
                else:
                    if self.models:
                        self.model_combo.current(0)
            else:
                self.model_combo.current(0)

            self.on_model_change(None)

        # 恢复深色模式
        if self.config.get("dark_mode", False):
            self.toggle_dark_mode()

    def on_model_change(self, event):
        """模型选择改变事件"""
        selection = self.model_combo.current()
        if selection >= 0 and selection < len(self.models):
            model = self.models[selection]

            # 显示文件大小和模型信息
            info_text = f"大小: {model['size_mb']:.1f} MB"
            if model["has_multimodal"]:
                info_text += "  |  ✅ 支持多模态"
            else:
                info_text += "  |  ❌ 不支持多模态"

            self.model_info_label.config(text=info_text)

            if model["has_multimodal"]:
                self.multimodal_check.config(state=tk.NORMAL)
                self.multimodal_info.config(
                    text=f"投影文件: {Path(model['mmproj_path']).name}",
                    foreground="green"
                )
            else:
                self.multimodal_var.set(False)
                self.multimodal_check.config(state=tk.DISABLED)
                self.multimodal_info.config(
                    text="该模型没有多模态投影文件",
                    foreground="gray"
                )

            self.update_server_frame_visibility()

    def on_mode_change(self):
        """启动模式改变事件"""
        self.update_server_frame_visibility()
        mode = self.mode_var.get()
        if mode == "server":
            self.check_port_status()

    def update_server_frame_visibility(self):
        """更新服务器配置框架的可见性"""
        mode = self.mode_var.get()
        if mode == "server":
            self.server_frame.grid()
        else:
            self.server_frame.grid_remove()
            self.webui_button.config(state=tk.DISABLED)

    def on_multimodal_change(self):
        """多模态选项改变事件"""
        pass

    def refresh_models(self):
        """刷新模型列表"""
        self.models = self.scan_models()
        self.model_combo["values"] = [m["display_name"] for m in self.models]
        if self.models:
            self.model_combo.current(0)
            self.on_model_change(None)
        self.log("模型列表已刷新", "info")

    # ======================== 端口检测 ========================

    def check_port_status(self):
        """检测 llama-server 是否真正就绪（HTTP 200 响应）"""
        def _check():
            host = self.host_var.get()
            port = self.port_var.get()
            url = f"http://{host}:{port}/health"
            try:
                import urllib.request
                req = urllib.request.Request(url, method='HEAD')
                resp = urllib.request.urlopen(req, timeout=3)
                if resp.status == 200:
                    self.root.after(0, lambda: self.port_status_label.config(
                        text=f"✅ 端口 {port} 服务就绪", foreground="green"))
                    self.root.after(0, lambda: self.webui_button.config(state=tk.NORMAL))
                else:
                    self.root.after(0, lambda: self.port_status_label.config(
                        text=f"端口 {port} 启动中... ({resp.status})", foreground="orange"))
                    self.root.after(0, lambda: self.webui_button.config(state=tk.DISABLED))
            except Exception:
                self.root.after(0, lambda: self.port_status_label.config(
                    text=f"端口 {port} 等待启动...", foreground="gray"))
                self.root.after(0, lambda: self.webui_button.config(state=tk.DISABLED))

        threading.Thread(target=_check, daemon=True).start()

    def open_webui(self):
        """打开 Web UI"""
        host = self.host_var.get()
        port = self.port_var.get()
        url = f"http://{host}:{port}"
        try:
            import webbrowser
            webbrowser.open(url)
            self.log(f"已打开 Web UI: {url}", "info")
        except Exception as e:
            self.log(f"打开 Web UI 失败: {e}", "error")

    # ======================== 深色模式 ========================

    def toggle_dark_mode(self, apply_only=False):
        """切换深色/浅色模式"""
        if not apply_only:
            self.dark_mode = not self.dark_mode
            self.dark_btn.config(text="☀️ 浅色模式" if self.dark_mode else "🌙 深色模式")

        bg = "#1e1e1e" if self.dark_mode else "#ffffff"
        fg = "#d4d4d4" if self.dark_mode else "#000000"
        insert_bg = "#ffffff" if self.dark_mode else "#000000"

        self.log_text.config(bg=bg, fg=fg, insertbackground=insert_bg)

        # 更新错误/警告颜色以适应深色模式
        if self.dark_mode:
            self.log_text.tag_config("error", foreground="#ff6b6b")
            self.log_text.tag_config("warning", foreground="#ffa726")
            self.log_text.tag_config("info", foreground="#64b5f6")
            self.log_text.tag_config("success", foreground="#81c784")
        else:
            self.log_text.tag_config("error", foreground="#ff0000")
            self.log_text.tag_config("warning", foreground="#ff8c00")
            self.log_text.tag_config("info", foreground="#0000ff")
            self.log_text.tag_config("success", foreground="#008000")

    # ======================== 命令生成 ========================

    def generate_command(self) -> Tuple[str, List[str]]:
        """生成启动命令"""
        selection = self.model_combo.current()
        if selection < 0 or selection >= len(self.models):
            raise ValueError("请选择一个模型")

        model = self.models[selection]
        mode = self.mode_var.get()

        exe_map = {
            "server": "llama-server.exe",
            "cli": "llama-cli.exe",
            "completion": "llama-completion.exe",
            "bench": "llama-bench.exe"
        }

        exe_name = exe_map.get(mode, "llama-server.exe")
        exe_path = Path(__file__).parent / exe_name

        if not exe_path.exists():
            raise FileNotFoundError(f"找不到可执行文件: {exe_name}")

        args = [str(exe_path)]
        args.extend(["-m", model["path"]])

        # GPU 参数
        ngl = self.ngl_var.get()
        if ngl:
            if mode == "bench":
                if ngl in ["auto", "all"]:
                    ngl_value = "99"
                else:
                    ngl_value = str(ngl)
                args.extend(["-ngl", ngl_value])
            else:
                args.extend(["-ngl", str(ngl)])

        # 线程参数
        threads = self.threads_var.get()
        if threads != -1:
            args.extend(["-t", str(threads)])

        # 上下文大小
        ctx_size = self.ctx_size_var.get()
        if ctx_size > 0 and mode != "bench":
            args.extend(["-c", str(ctx_size)])

        # 批处理大小
        batch_size = self.batch_size_var.get()
        if batch_size > 0:
            args.extend(["-b", str(batch_size)])

        # 微批处理大小
        ubatch_size = self.ubatch_size_var.get()
        if ubatch_size > 0:
            args.extend(["-ub", str(ubatch_size)])

        # Flash Attention
        flash_attn = self.flash_attn_var.get()
        if flash_attn and flash_attn != "auto":
            if mode == "bench":
                flash_value = "1" if flash_attn == "on" else "0"
                args.extend(["-fa", flash_value])
            else:
                args.extend(["-fa", flash_attn])

        # KV Cache 类型
        if mode != "bench":
            kv_cache = self.kv_cache_var.get()
            if kv_cache and kv_cache != "f16":
                args.extend(["-ctk", kv_cache])
                args.extend(["-ctv", kv_cache])

        # 多模态支持
        if mode in ["server", "cli"]:
            if self.multimodal_var.get() and model["has_multimodal"]:
                args.extend(["--mmproj", model["mmproj_path"]])

        # 服务器特定参数
        if mode == "server":
            host = self.host_var.get()
            port = self.port_var.get()
            args.extend(["--host", host])
            args.extend(["--port", str(port)])

        # 高级选项
        if mode != "bench":
            if self.mlock_var.get():
                args.append("--mlock")
            if not self.mmap_var.get():
                args.append("--no-mmap")
            if self.embedding_var.get():
                args.append("--embedding")
            if self.tools_var.get():
                args.append("--tools")
                args.append("all")

        return str(exe_path), args[1:]

    # ======================== 进程管理 ========================

    def start_process(self):
        """启动llama进程（含启动前检查）"""
        try:
            # 检查是否已有进程在运行
            if self.process and self.process.poll() is None:
                messagebox.showwarning("警告", "已有进程在运行中，请先停止当前进程。")
                return

            # 检查模型选择
            selection = self.model_combo.current()
            if selection < 0 or selection >= len(self.models):
                messagebox.showerror("错误", "请先选择一个模型！")
                return

            model = self.models[selection]
            if not Path(model["path"]).exists():
                messagebox.showerror("错误", f"模型文件不存在:\n{model['path']}")
                return

            # 检查端口占用 (server 模式)
            mode = self.mode_var.get()
            if mode == "server":
                host = self.host_var.get()
                port = self.port_var.get()
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    result = s.connect_ex((host, port))
                    if result == 0:
                        s.close()
                        overwrite = messagebox.askyesno(
                            "端口占用",
                            f"端口 {port} 已被占用。\n是否继续启动？（可能会启动失败）"
                        )
                        if not overwrite:
                            return
                    s.close()
                except Exception:
                    pass

            # 生成命令
            exe_path, args = self.generate_command()

            self.log("=" * 60, "info")
            self.log(f"启动命令: {exe_path}", "info")
            self.log(f"参数: {' '.join(args)}", "info")
            self.log("=" * 60, "info")

            work_dir = Path(__file__).parent

            # 使用 CREATE_NO_WINDOW 避免弹出额外控制台窗口 (Windows)
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.process = subprocess.Popen(
                [exe_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(work_dir),
                bufsize=0,
                startupinfo=startupinfo
            )

            # 更新UI状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set(f"运行中 - PID: {self.process.pid}")
            self.notebook.select(1)  # 切换到高级参数页，方便查看日志

            # 启动输出读取线程
            self.process_thread = threading.Thread(
                target=self.read_process_output,
                daemon=True
            )
            self.process_thread.start()

            # 启动进程监控线程
            self.monitor_running = True
            self.monitor_thread = threading.Thread(
                target=self.monitor_process,
                daemon=True
            )
            self.monitor_thread.start()

            self.log(f"进程已启动 (PID: {self.process.pid})", "success")

        except Exception as e:
            messagebox.showerror("启动错误", f"启动进程失败:\n{e}")
            self.log(f"启动失败: {e}", "error")

    def read_process_output(self):
        """读取进程输出（支持颜色高亮，自动检测 server 就绪）"""
        try:
            for line in self.process.stdout:
                try:
                    decoded_line = line.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        decoded_line = line.decode('gbk')
                    except UnicodeDecodeError:
                        try:
                            decoded_line = line.decode('latin-1')
                        except UnicodeDecodeError:
                            decoded_line = str(line)

                decoded_line = decoded_line.rstrip()

                # server 就绪检测：日志里出现 http:// 地址就亮按钮
                if "http://" in decoded_line:
                    self.root.after(0, lambda: self.webui_button.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.port_status_label.config(
                        text=f"✅ 服务就绪", foreground="green"))

                # 根据内容判断日志级别
                lower = decoded_line.lower()
                if "error" in lower or "错误" in lower or "failed" in lower:
                    self.root.after(0, self.log, decoded_line, "error")
                elif "warn" in lower or "警告" in lower:
                    self.root.after(0, self.log, decoded_line, "warning")
                elif "info" in lower or "加载" in lower or "load" in lower:
                    self.root.after(0, self.log, decoded_line, "info")
                else:
                    self.root.after(0, self.log, decoded_line)
        except Exception as e:
            self.root.after(0, self.log, f"读取输出错误: {e}", "error")
        finally:
            self.root.after(0, self.on_process_end)

    def monitor_process(self):
        """监控进程状态，自动检测进程结束"""
        while self.monitor_running and self.process:
            if self.process.poll() is not None:
                self.root.after(0, self.on_process_end)
                break
            time.sleep(1)

    def on_process_end(self):
        """进程结束回调"""
        if self.process:
            return_code = self.process.poll()
            if return_code is not None:
                self.log("=" * 60)
                self.log(f"进程已结束，返回码: {return_code}", "warning" if return_code != 0 else "success")
                self.status_var.set(f"已停止 - 返回码: {return_code}")

        self.monitor_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.process = None

        # 更新端口状态
        if self.mode_var.get() == "server":
            self.check_port_status()

    def stop_process(self):
        """停止llama进程"""
        if self.process:
            try:
                self.log("正在停止进程...", "warning")
                self.monitor_running = False

                # 先尝试优雅终止
                self.process.terminate()

                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.log("进程未响应，强制终止...", "warning")
                    self.process.kill()
                    self.process.wait()

                self.log("进程已停止", "success")
                self.status_var.set("已停止")

            except Exception as e:
                self.log(f"停止进程失败: {e}", "error")
            finally:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.process = None

    # ======================== 日志 ========================

    def log(self, message: str, level: str = None):
        """添加日志消息（支持颜色级别）"""
        if level:
            self.log_text.insert(tk.END, message + "\n", level)
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        # 限制日志行数，防止内存溢出
        max_lines = 5000
        current_lines = int(self.log_text.index('end-1c').split('.')[0])
        if current_lines > max_lines:
            self.log_text.delete(1.0, f"{current_lines - max_lines}.0")

    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)

    def copy_log(self):
        """复制日志到剪贴板"""
        try:
            content = self.log_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.log("日志已复制到剪贴板", "info")
        except Exception as e:
            self.log(f"复制失败: {e}", "error")

    # ======================== 关闭处理 ========================

    def on_closing(self):
        """窗口关闭事件"""
        if self.process and self.process.poll() is None:
            if messagebox.askyesno("确认退出", "当前有进程正在运行，确定要退出吗？"):
                self.stop_process()
                self.monitor_running = False
                self.root.destroy()
        else:
            self.monitor_running = False
            self.root.destroy()


def main():
    """主函数"""
    root = tk.Tk()

    style = ttk.Style()
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")

    app = LlamaLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
