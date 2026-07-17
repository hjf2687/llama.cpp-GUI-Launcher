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


class ScrollableFrame(ttk.Frame):
    """可滚动的框架容器"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # 创建 Canvas 和滚动条
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)

        # inner_frame 大小变化时更新 canvas 滚动区域
        self.inner_frame.bind("<Configure>", self._on_frame_configure)

        # 创建窗口
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Canvas 配置
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 绑定鼠标滚轮
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Canvas 宽度变化时同步 inner_frame 宽度
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_frame_configure(self, event):
        """inner_frame 大小变化时更新滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Canvas 宽度变化时同步 inner_frame 宽度"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        """绑定鼠标滚轮"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        """解绑鼠标滚轮"""
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


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

        # Embedding 服务器进程
        self.process_emb = None
        self.process_emb_thread = None
        self.monitor_emb_thread = None
        self.monitor_emb_running = False

        # 主题状态
        self.dark_mode = False

        # 配置文件路径
        self.config_file = Path(__file__).parent / "launcher_config.json"
        self.config = self.load_config()

        # GPU 信息缓存
        self.gpu_info = self.detect_gpu()

        # 扫描模型 + 合并手动添加的模型
        self.models = self.scan_models()
        self.load_manual_models()

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
            "llama_cpp_path": str(Path(__file__).parent),
            "last_model": "",
            "last_mode": "server",
            "gpu_layers": "auto",
            "threads": -1,
            "threads_batch": -1,
            "ctx_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "host": "127.0.0.1",
            "port": 8080,
            "api_key": "",
            "flash_attention": "auto",
            "multimodal_enabled": False,
            "kv_cache_type": "f16",
            "mlock": False,
            "mmap": True,
            "dark_mode": False,
            "manual_folders": [],
            "embedding_model": "",
            "embedding_port": 8081,
            # 采样参数
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            # 其他参数
            "seed": -1,
            "system_prompt": "",
            "chat_template": "",
            "jinja": False,
            "n_cpu_moe": -1,
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
            self.config["llama_cpp_path"] = self.llama_path_var.get()
            self.config["last_model"] = self.model_var.get()
            self.config["last_mode"] = self.mode_var.get()
            self.config["gpu_layers"] = self.ngl_var.get()
            self.config["threads"] = self.threads_var.get()
            self.config["threads_batch"] = self.threads_batch_var.get()
            self.config["ctx_size"] = self.ctx_size_var.get()
            self.config["batch_size"] = self.batch_size_var.get()
            self.config["ubatch_size"] = self.ubatch_size_var.get()
            self.config["host"] = self.host_var.get()
            self.config["port"] = self.port_var.get()
            self.config["api_key"] = self.api_key_var.get()
            self.config["flash_attention"] = self.flash_attn_var.get()
            self.config["multimodal_enabled"] = self.multimodal_var.get()
            self.config["kv_cache_type"] = self.kv_cache_var.get()
            self.config["n_cpu_moe"] = self.n_cpu_moe_var.get()
            self.config["mlock"] = self.mlock_var.get()
            self.config["mmap"] = self.mmap_var.get()
            self.config["tools"] = self.tools_var.get()
            self.config["jinja"] = self.jinja_var.get()
            self.config["seed"] = self.seed_var.get()
            self.config["temperature"] = self.temp_var.get()
            self.config["top_p"] = self.top_p_var.get()
            self.config["top_k"] = self.top_k_var.get()
            self.config["min_p"] = self.min_p_var.get()
            self.config["repeat_penalty"] = self.repeat_penalty_var.get()
            self.config["frequency_penalty"] = self.freq_penalty_var.get()
            self.config["presence_penalty"] = self.presence_penalty_var.get()
            self.config["system_prompt"] = self.system_prompt_var.get()
            self.config["chat_template"] = self.chat_template_var.get()
            self.config["embedding_model"] = self.emb_model_var.get()
            self.config["embedding_port"] = self.emb_port_var.get()
            self.config["dark_mode"] = self.dark_mode
            manual_folders = list(set(
                str(Path(m["path"]).parent) for m in self.models if m.get("manual")
            ))
            self.config["manual_folders"] = manual_folders

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
            base_name = self._model_base_name(model_dir.stem)

            # 先尝试精确匹配文件名
            for pattern in [
                f"mmproj-{model_dir.stem}.gguf",
                f"mmproj-{base_name}.gguf",
            ]:
                candidate = model_dir.parent / pattern
                if candidate.exists():
                    mmproj_file = candidate
                    break

            # 再遍历同目录下所有 mmproj 比较核心名称
            if mmproj_file is None:
                for f in model_dir.parent.glob("mmproj-*.gguf"):
                    mmproj_base = self._model_base_name(f.stem.replace("mmproj-", ""))
                    if base_name and mmproj_base and base_name == mmproj_base:
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

    # ======================== 手动模型管理 ========================

    @staticmethod
    def _model_base_name(stem: str) -> str:
        """去掉模型文件名中的量化/精度后缀，提取核心名称"""
        return re.sub(
            r'[-_]?(Q[2-8]_[A-Z0-9_]+|IQ[0-9]+_[A-Z0-9_]+|[FBI]?16|FP32|BF16)$',
            '', stem, flags=re.IGNORECASE
        )

    def _scan_single_folder(self, folder: Path) -> List[Dict]:
        """扫描文件夹（递归）中的模型，自动匹配 mmproj"""
        models = []
        gguf_files = list(folder.rglob("*.gguf"))

        # 预先收集所有 mmproj 文件，按所在目录分组
        mmproj_by_dir = {}
        for f in gguf_files:
            if "mmproj" in f.name.lower():
                mmproj_by_dir.setdefault(f.parent, []).append(f)

        for gguf in gguf_files:
            # 跳过 mmproj 文件本身
            if "mmproj" in gguf.name.lower():
                continue

            # 显示相对路径，让用户知道在哪个子目录
            try:
                rel = gguf.relative_to(folder)
                display = f"📁 {rel}"
            except ValueError:
                display = f"📁 {gguf.stem}"

            model_info = {
                "name": gguf.stem,
                "path": str(gguf),
                "size_mb": gguf.stat().st_size / (1024 * 1024),
                "has_multimodal": False,
                "mmproj_path": None,
                "display_name": display,
                "manual": True
            }

            # 去掉量化后缀，取核心名称用于匹配
            model_base = self._model_base_name(gguf.stem)

            # 在同目录下查找 mmproj
            for f in mmproj_by_dir.get(gguf.parent, []):
                mmproj_base = self._model_base_name(f.stem.replace("mmproj-", ""))
                if model_base and mmproj_base and model_base == mmproj_base:
                    model_info["has_multimodal"] = True
                    model_info["mmproj_path"] = str(f)
                    break

            models.append(model_info)

        return models

    def load_manual_models(self):
        """从配置中加载手动添加的文件夹，扫描其中的模型"""
        manual_dirs = self.config.get("manual_folders", [])
        for dir_path in manual_dirs:
            p = Path(dir_path)
            if p.exists() and p.is_dir():
                found = self._scan_single_folder(p)
                self.models.extend(found)
        self.models.sort(key=lambda x: x["size_mb"], reverse=True)

    def save_manual_models(self):
        """保存手动添加的文件夹路径到配置"""
        folders = list(set(
            Path(m["path"]).parent for m in self.models if m.get("manual")
        ))
        self.config["manual_folders"] = [str(f) for f in folders]
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存手动模型配置失败: {e}")

    def add_manual_model(self):
        """通过文件夹对话框添加模型目录"""
        initial_dir = str(Path(__file__).parent / "models")
        if not Path(initial_dir).exists():
            initial_dir = str(Path(__file__).parent)

        folder_path = filedialog.askdirectory(
            title="选择包含 GGUF 模型的文件夹",
            initialdir=initial_dir
        )

        if not folder_path:
            return

        folder = Path(folder_path)

        # 检查是否已添加过这个文件夹
        existing_folders = set()
        for m in self.models:
            if m.get("manual"):
                existing_folders.add(str(Path(m["path"]).parent))

        if str(folder) in existing_folders:
            self.log(f"该文件夹已在列表中: {folder.name}", "info")
            return

        # 扫描文件夹
        found = self._scan_single_folder(folder)

        if not found:
            self.log(f"该文件夹中未找到 .gguf 模型: {folder.name}", "warning")
            return

        # 添加到模型列表
        self.models.extend(found)
        self.models.sort(key=lambda x: x["size_mb"], reverse=True)

        # 刷新下拉列表
        self.model_combo["values"] = [m["display_name"] for m in self.models]

        # 选中第一个新添加的模型
        for i, m in enumerate(self.models):
            if m["path"] == found[0]["path"]:
                self.model_combo.current(i)
                break

        self.on_model_change(None)
        self.save_manual_models()
        self.log(f"已添加 {len(found)} 个模型: {folder.name}", "success")

    def add_embedding_file(self):
        """通过文件对话框添加 Embedding 模型文件"""
        filetypes = [("GGUF 模型", "*.gguf"), ("所有文件", "*.*")]
        initial_dir = str(Path(__file__).parent / "models")
        if not Path(initial_dir).exists():
            initial_dir = str(Path(__file__).parent)

        file_path = filedialog.askopenfilename(
            title="选择 Embedding 模型文件",
            filetypes=filetypes,
            initialdir=initial_dir
        )

        if not file_path:
            return

        p = Path(file_path)
        display_name = f"📁 {p.parent.name}/{p.stem}"

        # 检查是否已在列表中
        for m in self._emb_models_list:
            if m["path"] == file_path:
                # 已存在，直接选中
                idx = self._emb_models_list.index(m)
                self.emb_model_combo.current(idx)
                self.log(f"该 Embedding 模型已在列表中: {p.name}", "info")
                return

        # 添加到 embedding 模型列表
        model_info = {
            "name": p.stem,
            "path": str(p),
            "size_mb": p.stat().st_size / (1024 * 1024),
            "has_multimodal": False,
            "mmproj_path": None,
            "display_name": display_name,
            "manual": True
        }
        self._emb_models_list.append(model_info)
        self.emb_model_combo["values"] = [m["display_name"] for m in self._emb_models_list]

        # 选中新添加的
        idx = self._emb_models_list.index(model_info)
        self.emb_model_combo.current(idx)
        self.log(f"已添加 Embedding 模型: {p.name}", "success")

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

        # 分页2: 高级参数（可滚动）
        advanced_tab = ScrollableFrame(self.notebook)
        self.notebook.add(advanced_tab, text="高级参数")
        self.create_advanced_config_widgets(advanced_tab.inner_frame)

        # 分页3: Embedding
        emb_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(emb_tab, text="Embedding")
        self.create_embedding_widgets(emb_tab)

        # 分页4: 硬件信息
        hw_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(hw_tab, text="硬件信息")
        self.create_hardware_widgets(hw_tab)

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

        # llama.cpp 路径
        path_frame = ttk.LabelFrame(parent, text="llama.cpp 路径", padding="8")
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        path_frame.columnconfigure(1, weight=1)

        self.llama_path_var = tk.StringVar(value=self.config.get("llama_cpp_path", str(Path(__file__).parent)))
        path_entry = ttk.Entry(path_frame, textvariable=self.llama_path_var, width=40)
        path_entry.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(
            path_frame,
            text="📂",
            command=self.browse_llama_path,
            width=3
        ).grid(row=0, column=2)

        # 验证路径
        self.path_status_label = ttk.Label(path_frame, text="", foreground="gray")
        self.path_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))
        self._validate_llama_path()

        # 模型选择
        model_frame = ttk.LabelFrame(parent, text="模型选择", padding="8")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        model_frame.columnconfigure(0, weight=1)

        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[m["display_name"] for m in self.models],
            state="readonly",
            width=38
        )
        self.model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        browse_btn = ttk.Button(
            model_frame,
            text="📂 添加目录...",
            command=self.add_manual_model,
            width=12
        )
        browse_btn.grid(row=1, column=1, padx=(5, 0), pady=(5, 0))

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
        mode_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

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
        gpu_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

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
        self.server_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

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

        # API 密钥
        ttk.Label(self.server_frame, text="API 密钥:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.api_key_var = tk.StringVar(value=self.config["api_key"])
        api_key_entry = ttk.Entry(self.server_frame, textvariable=self.api_key_var, width=25, show="*")
        api_key_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 一键打开 Web UI 按钮 (仅 server 模式)
        self.webui_button = ttk.Button(
            self.server_frame,
            text="🌐 打开 Web UI",
            command=self.open_webui,
            state=tk.DISABLED
        )
        self.webui_button.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # 端口检测放到后台，避免界面卡住
        self.root.after(100, self.check_port_status)

    def create_advanced_config_widgets(self, parent):
        """创建高级参数配置组件"""
        parent.columnconfigure(0, weight=1)

        # ===== 性能参数 =====
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

        # 批处理线程数
        ttk.Label(params_frame, text="批处理线程 (-ts):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.threads_batch_var = tk.IntVar(value=self.config["threads_batch"])
        threads_batch_spin = ttk.Spinbox(
            params_frame,
            from_=-1,
            to=64,
            textvariable=self.threads_batch_var,
            width=12
        )
        threads_batch_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(params_frame, text="(-1=自动)").grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # 上下文大小
        ttk.Label(params_frame, text="上下文大小 (-c):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.ctx_size_var = tk.IntVar(value=self.config["ctx_size"])
        ctx_spin = ttk.Spinbox(
            params_frame,
            from_=512,
            to=131072,
            increment=1024,
            textvariable=self.ctx_size_var,
            width=12
        )
        ctx_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 批处理大小
        ttk.Label(params_frame, text="批处理大小 (-b):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=self.config["batch_size"])
        batch_spin = ttk.Spinbox(
            params_frame,
            from_=64,
            to=8192,
            increment=64,
            textvariable=self.batch_size_var,
            width=12
        )
        batch_spin.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 微批处理大小
        ttk.Label(params_frame, text="微批处理大小 (-ub):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.ubatch_size_var = tk.IntVar(value=self.config["ubatch_size"])
        ubatch_spin = ttk.Spinbox(
            params_frame,
            from_=16,
            to=4096,
            increment=16,
            textvariable=self.ubatch_size_var,
            width=12
        )
        ubatch_spin.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Flash Attention
        ttk.Label(params_frame, text="Flash Attention (-fa):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.flash_attn_var = tk.StringVar(value=self.config["flash_attention"])
        flash_combo = ttk.Combobox(
            params_frame,
            textvariable=self.flash_attn_var,
            values=["auto", "on", "off"],
            state="readonly",
            width=12
        )
        flash_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # KV Cache 类型
        ttk.Label(params_frame, text="KV Cache 类型:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.kv_cache_var = tk.StringVar(value=self.config["kv_cache_type"])
        kv_combo = ttk.Combobox(
            params_frame,
            textvariable=self.kv_cache_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            state="readonly",
            width=12
        )
        kv_combo.grid(row=6, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # CPU MoE 层数
        ttk.Label(params_frame, text="CPU MoE 层数:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.n_cpu_moe_var = tk.IntVar(value=self.config.get("n_cpu_moe", -1))
        n_cpu_moe_spin = ttk.Spinbox(
            params_frame,
            from_=-1,
            to=128,
            textvariable=self.n_cpu_moe_var,
            width=12
        )
        n_cpu_moe_spin.grid(row=7, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(params_frame, text="(-1=自动, MoE模型专用)").grid(row=7, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # ===== 采样参数 =====
        sampling_frame = ttk.LabelFrame(parent, text="采样参数", padding="8")
        sampling_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # 温度
        ttk.Label(sampling_frame, text="温度 (--temp):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.temp_var = tk.DoubleVar(value=self.config["temperature"])
        temp_spin = ttk.Spinbox(
            sampling_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.temp_var,
            width=12,
            format="%.2f"
        )
        temp_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(sampling_frame, text="(0=精确, 2=随机)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # Top-p
        ttk.Label(sampling_frame, text="Top-p:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.top_p_var = tk.DoubleVar(value=self.config["top_p"])
        top_p_spin = ttk.Spinbox(
            sampling_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.top_p_var,
            width=12,
            format="%.2f"
        )
        top_p_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Top-k
        ttk.Label(sampling_frame, text="Top-k:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.top_k_var = tk.IntVar(value=self.config["top_k"])
        top_k_spin = ttk.Spinbox(
            sampling_frame,
            from_=1,
            to=200,
            textvariable=self.top_k_var,
            width=12
        )
        top_k_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Min-p
        ttk.Label(sampling_frame, text="Min-p:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.min_p_var = tk.DoubleVar(value=self.config["min_p"])
        min_p_spin = ttk.Spinbox(
            sampling_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.min_p_var,
            width=12,
            format="%.2f"
        )
        min_p_spin.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 重复惩罚
        ttk.Label(sampling_frame, text="重复惩罚:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.repeat_penalty_var = tk.DoubleVar(value=self.config["repeat_penalty"])
        repeat_spin = ttk.Spinbox(
            sampling_frame,
            from_=1.0,
            to=2.0,
            increment=0.1,
            textvariable=self.repeat_penalty_var,
            width=12,
            format="%.2f"
        )
        repeat_spin.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(sampling_frame, text="(1.0=关闭)").grid(row=4, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # 频率惩罚
        ttk.Label(sampling_frame, text="频率惩罚:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.freq_penalty_var = tk.DoubleVar(value=self.config["frequency_penalty"])
        freq_spin = ttk.Spinbox(
            sampling_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.freq_penalty_var,
            width=12,
            format="%.2f"
        )
        freq_spin.grid(row=5, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # 存在惩罚
        ttk.Label(sampling_frame, text="存在惩罚:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.presence_penalty_var = tk.DoubleVar(value=self.config["presence_penalty"])
        presence_spin = ttk.Spinbox(
            sampling_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.presence_penalty_var,
            width=12,
            format="%.2f"
        )
        presence_spin.grid(row=6, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # ===== 多模态支持 =====
        multimodal_frame = ttk.LabelFrame(parent, text="多模态支持", padding="8")
        multimodal_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

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

        # ===== 高级选项 =====
        advanced_frame = ttk.LabelFrame(parent, text="高级选项", padding="8")
        advanced_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

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

        self.tools_var = tk.BooleanVar(value=self.config.get("tools", False))
        ttk.Checkbutton(
            advanced_frame,
            text="内置工具 (--tools, 启用 function calling)",
            variable=self.tools_var
        ).grid(row=2, column=0, sticky=tk.W, pady=2)

        self.jinja_var = tk.BooleanVar(value=self.config.get("jinja", False))
        ttk.Checkbutton(
            advanced_frame,
            text="Jinja2 模板 (--jinja, 新模型兼容)",
            variable=self.jinja_var
        ).grid(row=3, column=0, sticky=tk.W, pady=2)

        # 随机种子
        ttk.Label(advanced_frame, text="随机种子 (-s):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.IntVar(value=self.config["seed"])
        seed_spin = ttk.Spinbox(
            advanced_frame,
            from_=-1,
            to=999999999,
            textvariable=self.seed_var,
            width=12
        )
        seed_spin.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(advanced_frame, text="(-1=随机)").grid(row=4, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # ===== 系统提示词 & 聊天模板 =====
        prompt_frame = ttk.LabelFrame(parent, text="系统提示词 & 聊天模板", padding="8")
        prompt_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        prompt_frame.columnconfigure(1, weight=1)

        # 系统提示词
        ttk.Label(prompt_frame, text="系统提示词:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.system_prompt_var = tk.StringVar(value=self.config.get("system_prompt", ""))
        system_prompt_entry = ttk.Entry(
            prompt_frame,
            textvariable=self.system_prompt_var,
            width=50
        )
        system_prompt_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        ttk.Label(prompt_frame, text="(-sp 参数)", foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=(5, 0), pady=2)

        # 聊天模板文件
        ttk.Label(prompt_frame, text="聊天模板文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.chat_template_var = tk.StringVar(value=self.config.get("chat_template", ""))
        chat_template_entry = ttk.Entry(
            prompt_frame,
            textvariable=self.chat_template_var,
            width=50
        )
        chat_template_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)

        # 浏览按钮
        def browse_chat_template():
            path = filedialog.askopenfilename(
                title="选择聊天模板文件",
                filetypes=[("Jinja 模板", "*.jinja *.j2"), ("所有文件", "*.*")]
            )
            if path:
                self.chat_template_var.set(path)

        ttk.Button(
            prompt_frame,
            text="📂",
            command=browse_chat_template,
            width=3
        ).grid(row=1, column=2, padx=(5, 0), pady=2)

    def create_embedding_widgets(self, parent):
        """创建 Embedding 配置组件"""
        parent.columnconfigure(0, weight=1)

        # 模型选择
        model_frame = ttk.LabelFrame(parent, text="Embedding 模型", padding="8")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        model_frame.columnconfigure(0, weight=1)

        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W)

        emb_models = [m for m in self.models if "embedding" in m["name"].lower()]
        self._emb_models_list = emb_models
        self.emb_model_var = tk.StringVar(value=self.config.get("embedding_model", ""))
        self.emb_model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.emb_model_var,
            values=[m["display_name"] for m in emb_models],
            state="readonly",
            width=40
        )
        self.emb_model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        emb_browse_btn = ttk.Button(
            model_frame,
            text="📂 选择文件...",
            command=self.add_embedding_file,
            width=12
        )
        emb_browse_btn.grid(row=1, column=1, padx=(5, 0), pady=(5, 0))

        self.emb_model_info_label = ttk.Label(model_frame, text="", wraplength=350)
        self.emb_model_info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # 绑定选择事件
        self.emb_model_combo.bind("<<ComboboxSelected>>", self.on_emb_model_change)

        # 端口配置
        port_frame = ttk.LabelFrame(parent, text="服务器配置", padding="8")
        port_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        ttk.Label(port_frame, text="端口:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.emb_port_var = tk.IntVar(value=self.config.get("embedding_port", 8081))
        self.emb_port_spin = ttk.Spinbox(
            port_frame,
            from_=1024,
            to=65535,
            textvariable=self.emb_port_var,
            width=12
        )
        self.emb_port_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(port_frame, text="(不能与主模型端口相同)").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0)
        )

        # 操作按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.emb_start_btn = ttk.Button(
            btn_frame,
            text="▶ 启动 Embedding",
            command=self.start_embedding_server
        )
        self.emb_start_btn.grid(row=0, column=0, padx=(0, 5))

        self.emb_stop_btn = ttk.Button(
            btn_frame,
            text="⏹ 停止",
            command=self.stop_embedding_server,
            state=tk.DISABLED
        )
        self.emb_stop_btn.grid(row=0, column=1)

        self.emb_status_var = tk.StringVar(value="未启动")
        ttk.Label(
            btn_frame,
            textvariable=self.emb_status_var,
            foreground="gray"
        ).grid(row=0, column=2, padx=(10, 0))

    def on_emb_model_change(self, event):
        """Embedding 模型选择改变"""
        selection = self.emb_model_combo.current()
        if selection >= 0 and selection < len(self._emb_models_list):
            model = self._emb_models_list[selection]
            info_text = f"大小: {model['size_mb']:.1f} MB  |  路径: {Path(model['path']).parent.name}"
            self.emb_model_info_label.config(text=info_text)

    def create_hardware_widgets(self, parent):
        """创建硬件信息组件"""
        import psutil

        parent.columnconfigure(0, weight=1)

        # ===== CPU =====
        cpu_frame = ttk.LabelFrame(parent, text="CPU", padding="8")
        cpu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        cpu_frame.columnconfigure(1, weight=1)

        cpu_info = psutil.cpu_freq()
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        cpu_logical = psutil.cpu_count(logical=True)

        # 型号
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except Exception:
            cpu_name = f"{cpu_count} 核心 CPU"
        ttk.Label(cpu_frame, text=cpu_name, wraplength=400).grid(row=0, column=0, columnspan=3, sticky=tk.W)

        freq_text = f"{cpu_count}C/{cpu_logical}T"
        if cpu_info:
            freq_text += f"  {cpu_info.current:.0f} MHz"
        ttk.Label(cpu_frame, text=freq_text, foreground="gray").grid(row=1, column=0, columnspan=3, sticky=tk.W)

        # CPU 使用率条
        ttk.Label(cpu_frame, text="使用率").grid(row=2, column=0, sticky=tk.W)
        self.cpu_bar = ttk.Progressbar(cpu_frame, length=200, mode='determinate', maximum=100)
        self.cpu_bar.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(8, 8))
        self.cpu_pct_label = ttk.Label(cpu_frame, text="0%", width=6)
        self.cpu_pct_label.grid(row=2, column=2, sticky=tk.E)

        # ===== 内存 =====
        mem_frame = ttk.LabelFrame(parent, text="内存 (RAM)", padding="8")
        mem_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        mem_frame.columnconfigure(1, weight=1)

        mem = psutil.virtual_memory()
        ttk.Label(mem_frame,
            text=f"{mem.total / (1024**3):.1f} GB").grid(row=0, column=0, columnspan=3, sticky=tk.W)

        ttk.Label(mem_frame, text="使用率").grid(row=1, column=0, sticky=tk.W)
        self.mem_bar = ttk.Progressbar(mem_frame, length=200, mode='determinate', maximum=100)
        self.mem_bar.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(8, 8))
        self.mem_pct_label = ttk.Label(mem_frame, text="0%", width=6)
        self.mem_pct_label.grid(row=1, column=2, sticky=tk.E)

        self.mem_detail_label = ttk.Label(mem_frame, text="", foreground="gray")
        self.mem_detail_label.grid(row=2, column=0, columnspan=3, sticky=tk.W)

        # ===== GPU =====
        gpu_frame = ttk.LabelFrame(parent, text="GPU", padding="8")
        gpu_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        gpu_frame.columnconfigure(1, weight=1)

        if self.gpu_info.get("cuda"):
            ttk.Label(gpu_frame, text=self.gpu_info["gpu_name"]).grid(
                row=0, column=0, columnspan=3, sticky=tk.W)
            ttk.Label(gpu_frame,
                text=f"{self.gpu_info['vram_mb']} MB 显存",
                foreground="gray").grid(row=1, column=0, columnspan=3, sticky=tk.W)

            # GPU 使用率条
            ttk.Label(gpu_frame, text="使用率").grid(row=2, column=0, sticky=tk.W)
            self.gpu_bar = ttk.Progressbar(gpu_frame, length=200, mode='determinate', maximum=100)
            self.gpu_bar.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(8, 8))
            self.gpu_pct_label = ttk.Label(gpu_frame, text="0%", width=6)
            self.gpu_pct_label.grid(row=2, column=2, sticky=tk.E)

            # GPU 显存条
            ttk.Label(gpu_frame, text="显存").grid(row=3, column=0, sticky=tk.W)
            self.gpu_mem_bar = ttk.Progressbar(gpu_frame, length=200, mode='determinate', maximum=100)
            self.gpu_mem_bar.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(8, 8))
            self.gpu_mem_pct_label = ttk.Label(gpu_frame, text="0%", width=6)
            self.gpu_mem_pct_label.grid(row=3, column=2, sticky=tk.E)
        else:
            ttk.Label(gpu_frame, text="未检测到 NVIDIA GPU", foreground="gray").grid(
                row=0, column=0, columnspan=3, sticky=tk.W)

        # ===== 磁盘 =====
        disk_frame = ttk.LabelFrame(parent, text="磁盘", padding="8")
        disk_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        disk_frame.columnconfigure(1, weight=1)

        llama_path = Path(self.config.get("llama_cpp_path", str(Path(__file__).parent)))
        try:
            disk = psutil.disk_usage(str(llama_path))
            ttk.Label(disk_frame,
                text=f"{llama_path.drive} 盘  {disk.used / (1024**3):.1f} / {disk.total / (1024**3):.1f} GB").grid(
                row=0, column=0, columnspan=3, sticky=tk.W)

            ttk.Label(disk_frame, text="使用率").grid(row=1, column=0, sticky=tk.W)
            self.disk_bar = ttk.Progressbar(disk_frame, length=200, mode='determinate', maximum=100)
            self.disk_bar.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(8, 8))
            self.disk_pct_label = ttk.Label(disk_frame, text="0%", width=6)
            self.disk_pct_label.grid(row=1, column=2, sticky=tk.E)
        except Exception:
            pass

        # ===== 刷新控制 =====
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(8, 0))

        ttk.Label(ctrl_frame, text="自动刷新:").grid(row=0, column=0, padx=(0, 5))

        self.hw_refresh_var = tk.StringVar(value="3")
        refresh_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.hw_refresh_var,
            values=["1", "2", "3", "5", "10"],
            state="readonly",
            width=4
        )
        refresh_combo.grid(row=0, column=1, padx=(0, 3))
        ttk.Label(ctrl_frame, text="秒").grid(row=0, column=2, padx=(0, 10))

        self.hw_autorefresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl_frame,
            text="启用",
            variable=self.hw_autorefresh_var,
            command=self._toggle_hw_autorefresh
        ).grid(row=0, column=3, padx=(0, 10))

        self.hw_timer_id = None
        self.refresh_hardware()
        self._start_hw_timer()

    def _start_hw_timer(self):
        """启动硬件信息定时刷新"""
        if self.hw_timer_id:
            self.root.after_cancel(self.hw_timer_id)
        if self.hw_autorefresh_var.get():
            try:
                interval = int(self.hw_refresh_var.get()) * 1000
            except ValueError:
                interval = 3000
            self.hw_timer_id = self.root.after(interval, self._hw_timer_tick)

    def _hw_timer_tick(self):
        """定时刷新回调"""
        self.refresh_hardware()
        self._start_hw_timer()

    def _toggle_hw_autorefresh(self):
        """切换自动刷新"""
        if self.hw_autorefresh_var.get():
            self._start_hw_timer()
        elif self.hw_timer_id:
            self.root.after_cancel(self.hw_timer_id)
            self.hw_timer_id = None

    def _get_usage_color(self, pct: float) -> str:
        """根据百分比返回颜色"""
        if pct >= 85:
            return "#ff4444"  # 红
        elif pct >= 60:
            return "#ffaa00"  # 黄
        else:
            return "#44bb44"  # 绿

    def refresh_hardware(self):
        """刷新硬件信息"""
        import psutil

        # CPU 使用率
        usage = psutil.cpu_percent(interval=0)
        self.cpu_bar["value"] = usage
        self.cpu_pct_label.config(text=f"{usage:.0f}%", foreground=self._get_usage_color(usage))

        # 内存
        mem = psutil.virtual_memory()
        self.mem_bar["value"] = mem.percent
        self.mem_pct_label.config(text=f"{mem.percent:.0f}%", foreground=self._get_usage_color(mem.percent))
        self.mem_detail_label.config(
            text=f"已用 {mem.used / (1024**3):.1f} GB / 可用 {mem.available / (1024**3):.1f} GB")

        # GPU
        if self.gpu_info.get("cuda"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        gpu_pct = float(parts[0])
                        mem_used = float(parts[1])
                        mem_total = float(parts[2])
                        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

                        self.gpu_bar["value"] = gpu_pct
                        self.gpu_pct_label.config(
                            text=f"{gpu_pct:.0f}%", foreground=self._get_usage_color(gpu_pct))
                        self.gpu_mem_bar["value"] = mem_pct
                        self.gpu_mem_pct_label.config(
                            text=f"{mem_pct:.0f}%", foreground=self._get_usage_color(mem_pct))
            except Exception:
                pass

        # 磁盘
        if hasattr(self, 'disk_bar'):
            llama_path = Path(self.config.get("llama_cpp_path", str(Path(__file__).parent)))
            try:
                disk = psutil.disk_usage(str(llama_path))
                self.disk_bar["value"] = disk.percent
                self.disk_pct_label.config(
                    text=f"{disk.percent:.0f}%", foreground=self._get_usage_color(disk.percent))
            except Exception:
                pass

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

    def browse_llama_path(self):
        """选择 llama.cpp 目录"""
        folder = filedialog.askdirectory(
            title="选择 llama.cpp 目录",
            initialdir=self.llama_path_var.get()
        )
        if folder:
            self.llama_path_var.set(folder)
            self._validate_llama_path()

    def _validate_llama_path(self):
        """验证 llama.cpp 路径是否有效"""
        path = Path(self.llama_path_var.get())
        exe = path / "llama-server.exe"
        if exe.exists():
            self.path_status_label.config(text=f"✅ 找到 llama-server.exe", foreground="green")
        else:
            self.path_status_label.config(text=f"⚠️ 未找到 llama-server.exe", foreground="orange")

    def get_llama_exe(self, name: str) -> Path:
        """获取 llama.cpp 可执行文件路径"""
        return Path(self.llama_path_var.get()) / name

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
        self.load_manual_models()
        self.model_combo["values"] = [m["display_name"] for m in self.models]
        if self.models:
            self.model_combo.current(0)
            self.on_model_change(None)
        # 刷新 embedding 模型列表
        self._emb_models_list = [m for m in self.models if "embedding" in m["name"].lower()]
        self.emb_model_combo["values"] = [m["display_name"] for m in self._emb_models_list]
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
        exe_path = self.get_llama_exe(exe_name)

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

        # 批处理线程数
        threads_batch = self.threads_batch_var.get()
        if threads_batch != -1:
            args.extend(["-ts", str(threads_batch)])

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

        # CPU MoE 层数 (MoE 模型专用)
        n_cpu_moe = self.n_cpu_moe_var.get()
        if n_cpu_moe != -1:
            args.extend(["--n-cpu-moe", str(n_cpu_moe)])

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

            # API 密钥
            api_key = self.api_key_var.get().strip()
            if api_key:
                args.extend(["--api-key", api_key])

        # 采样参数 (仅 cli/completion 模式)
        if mode in ("cli", "completion"):
            temp = self.temp_var.get()
            if temp != 0.8:  # 默认值
                args.extend(["--temp", f"{temp:.2f}"])

            top_p = self.top_p_var.get()
            if top_p != 0.9:
                args.extend(["--top-p", f"{top_p:.2f}"])

            top_k = self.top_k_var.get()
            if top_k != 40:
                args.extend(["--top-k", str(top_k)])

            min_p = self.min_p_var.get()
            if min_p != 0.05:
                args.extend(["--min-p", f"{min_p:.2f}"])

            repeat_penalty = self.repeat_penalty_var.get()
            if repeat_penalty != 1.1:
                args.extend(["--repeat-penalty", f"{repeat_penalty:.2f}"])

            freq_penalty = self.freq_penalty_var.get()
            if freq_penalty != 0.0:
                args.extend(["--frequency-penalty", f"{freq_penalty:.2f}"])

            presence_penalty = self.presence_penalty_var.get()
            if presence_penalty != 0.0:
                args.extend(["--presence-penalty", f"{presence_penalty:.2f}"])

        # 高级选项
        if mode != "bench":
            if self.mlock_var.get():
                args.append("--mlock")
            if not self.mmap_var.get():
                args.append("--no-mmap")
            if self.tools_var.get():
                args.append("--tools")
                args.append("all")
            if self.jinja_var.get():
                args.append("--jinja")

            # 随机种子
            seed = self.seed_var.get()
            if seed != -1:
                args.extend(["-s", str(seed)])

            # 系统提示词
            system_prompt = self.system_prompt_var.get().strip()
            if system_prompt:
                args.extend(["-sp", system_prompt])

            # 聊天模板
            chat_template = self.chat_template_var.get().strip()
            if chat_template:
                args.extend(["--chat-template-file", chat_template])

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

            work_dir = Path(self.llama_path_var.get())

            # 交互模式（cli/completion）弹出终端窗口，用户直接操作
            interactive = mode in ("cli", "completion")

            if interactive:
                # 写临时 bat，用 Popen + CREATE_NEW_CONSOLE 弹出终端
                bat_lines = ["@echo off", f'cd /d "{work_dir}"']
                quoted = [f'"{exe_path}"']
                for a in args:
                    a_str = str(a)
                    quoted.append(f'"{a_str}"' if " " in a_str else a_str)
                bat_lines.append(" ".join(quoted))
                bat_lines.append("pause")
                bat_lines.append("")
                bat_path = work_dir / "_launch_cli.bat"
                bat_path.write_text("\n".join(bat_lines), encoding="utf-8")

                # 用 Popen 启动 bat，CREATE_NEW_CONSOLE 打开新终端窗口
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = subprocess.CREATE_NEW_CONSOLE
                self.process = subprocess.Popen(
                    ["cmd.exe", "/c", str(bat_path)],
                    cwd=str(work_dir),
                    creationflags=creation_flags
                )
            else:
                # server/bench 模式：隐藏窗口，接管输出
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

            if interactive:
                self.status_var.set(f"运行中 - PID: {self.process.pid}")
                self.log("已弹出终端窗口，请在终端中操作", "info")
                self.log("点击「停止」或关闭终端窗口即可停止", "info")

                # 启动进程监控线程（检测终端关闭）
                self.monitor_running = True
                self.monitor_thread = threading.Thread(
                    target=self.monitor_process,
                    daemon=True
                )
                self.monitor_thread.start()
            else:
                self.status_var.set(f"运行中 - PID: {self.process.pid}")
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
        """主进程结束回调"""
        if self.process:
            return_code = self.process.poll()
            if return_code is not None:
                self.log("=" * 60)
                self.log(f"主进程已结束，返回码: {return_code}", "warning" if return_code != 0 else "success")
                self.monitor_running = False
                self.process = None

                # 清理临时 bat 文件
                try:
                    bat_path = Path(self.llama_path_var.get()) / "_launch_cli.bat"
                    if bat_path.exists():
                        bat_path.unlink()
                except:
                    pass

        # 如果 embedding 也在运行，不停止按钮
        if self.process_emb and self.process_emb.poll() is None:
            self.status_var.set(f"Embedding 运行中 - PID: {self.process_emb.pid}")
            return

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("已停止")

        # 更新端口状态
        if self.mode_var.get() == "server":
            self.check_port_status()

    def on_emb_process_end(self):
        """Embedding 进程结束回调"""
        if self.process_emb:
            return_code = self.process_emb.poll()
            if return_code is not None:
                self.log(f"Embedding 服务器已结束，返回码: {return_code}",
                         "warning" if return_code != 0 else "success")
                self.monitor_emb_running = False
                self.process_emb = None

        # 恢复 Embedding 按钮状态
        self.emb_start_btn.config(state=tk.NORMAL)
        self.emb_stop_btn.config(state=tk.DISABLED)
        self.emb_status_var.set("已停止")
        self.emb_model_combo.config(state=tk.NORMAL)
        self.emb_port_spin.config(state=tk.NORMAL)

        # 如果主进程也没在运行，恢复主按钮
        if not self.process or self.process.poll() is not None:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("已停止")
            if self.mode_var.get() == "server":
                self.check_port_status()
        else:
            self.status_var.set(f"运行中 - PID: {self.process.pid}")

    def stop_process(self):
        """停止所有 llama 进程"""
        # 先停 embedding
        self._stop_single_process(self.process_emb, "Embedding 服务器")
        self.process_emb = None
        self.monitor_emb_running = False

        # 再停主进程
        self._stop_single_process(self.process, "主进程")
        self.process = None
        self.monitor_running = False

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("已停止")

    def _stop_single_process(self, proc, name: str):
        """停止单个进程及其子进程"""
        if proc is None:
            return

        try:
            # 检查进程是否还在运行
            if proc.poll() is not None:
                self.log(f"{name} 已结束", "info")
                return

            self.log(f"正在停止 {name}...", "warning")
            pid = proc.pid

            # Windows 上用 taskkill /F /T 杀掉整个进程树
            if sys.platform == "win32":
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self.log(f"{name} 已停止", "success")
                else:
                    # 可能进程已结束
                    self.log(f"{name} 已结束", "info")
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self.log(f"{name} 已停止", "success")
        except Exception as e:
            self.log(f"停止 {name} 失败: {e}", "error")

    # ======================== Embedding 服务器 ========================

    def stop_embedding_server(self):
        """单独停止 Embedding 服务器"""
        self._stop_single_process(self.process_emb, "Embedding 服务器")
        self.process_emb = None
        self.monitor_emb_running = False

        # 恢复按钮状态
        self.emb_start_btn.config(state=tk.NORMAL)
        self.emb_stop_btn.config(state=tk.DISABLED)
        self.emb_status_var.set("已停止")
        self.emb_model_combo.config(state=tk.NORMAL)
        self.emb_port_spin.config(state=tk.NORMAL)

    def start_embedding_server(self):
        """启动独立的 Embedding 服务器"""
        try:
            # 找到选中的 embedding 模型
            emb_selection = self.emb_model_combo.current()
            if emb_selection < 0 or emb_selection >= len(self._emb_models_list):
                self.log("未选择 Embedding 模型，跳过启动", "warning")
                return

            emb_model = self._emb_models_list[emb_selection]
            if not Path(emb_model["path"]).exists():
                self.log(f"Embedding 模型文件不存在: {emb_model['path']}", "error")
                return

            host = self.host_var.get()
            emb_port = self.emb_port_var.get()

            # 检查端口是否与主进程冲突
            main_port = self.port_var.get()
            if emb_port == main_port:
                self.log("Embedding 端口不能与主模型端口相同！", "error")
                return

            exe_path = self.get_llama_exe("llama-server.exe")
            if not exe_path.exists():
                self.log("找不到 llama-server.exe，无法启动 Embedding 服务器", "error")
                return

            args = [
                str(exe_path),
                "-m", emb_model["path"],
                "--host", host,
                "--port", str(emb_port),
                "--embedding",
                "-c", "512",  # embedding 通常不需要大上下文
            ]

            # GPU 参数 - embedding 模型通常较小，用较少层数
            ngl = self.ngl_var.get()
            if ngl and ngl not in ["auto", "all"]:
                ngl_val = min(int(ngl), 20)  # embedding 模型限制 GPU 层数
                args.extend(["-ngl", str(ngl_val)])
            else:
                args.extend(["-ngl", "99"])

            self.log("=" * 60, "info")
            self.log(f"启动 Embedding 服务器: {emb_model['name']}", "info")
            self.log(f"参数: {' '.join(args[1:])}", "info")
            self.log("=" * 60, "info")

            work_dir = Path(self.llama_path_var.get())
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.process_emb = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(work_dir),
                bufsize=0,
                startupinfo=startupinfo
            )

            # 启动输出读取线程
            self.process_emb_thread = threading.Thread(
                target=self.read_embedding_output,
                daemon=True
            )
            self.process_emb_thread.start()

            # 启动监控线程
            self.monitor_emb_running = True
            self.monitor_emb_thread = threading.Thread(
                target=self.monitor_embedding_process,
                daemon=True
            )
            self.monitor_emb_thread.start()

            self.log(f"Embedding 服务器已启动 (PID: {self.process_emb.pid}, 端口: {emb_port})", "success")

            # 更新按钮状态
            self.emb_start_btn.config(state=tk.DISABLED)
            self.emb_stop_btn.config(state=tk.NORMAL)
            self.emb_status_var.set(f"运行中 - PID: {self.process_emb.pid}")
            self.emb_model_combo.config(state=tk.DISABLED)
            self.emb_port_spin.config(state=tk.DISABLED)

        except Exception as e:
            self.log(f"启动 Embedding 服务器失败: {e}", "error")

    def read_embedding_output(self):
        """读取 Embedding 服务器输出"""
        try:
            for line in self.process_emb.stdout:
                try:
                    decoded_line = line.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        decoded_line = line.decode('gbk')
                    except UnicodeDecodeError:
                        decoded_line = line.decode('latin-1', errors='replace')

                decoded_line = decoded_line.rstrip()
                # 加前缀区分来源
                lower = decoded_line.lower()
                if "error" in lower or "错误" in lower or "failed" in lower:
                    self.root.after(0, self.log, f"[Emb] {decoded_line}", "error")
                elif "warn" in lower or "警告" in lower:
                    self.root.after(0, self.log, f"[Emb] {decoded_line}", "warning")
                else:
                    self.root.after(0, self.log, f"[Emb] {decoded_line}")
        except Exception as e:
            self.root.after(0, self.log, f"[Emb] 读取输出错误: {e}", "error")
        finally:
            self.root.after(0, self.on_emb_process_end)

    def monitor_embedding_process(self):
        """监控 Embedding 进程状态"""
        while self.monitor_emb_running and self.process_emb:
            if self.process_emb.poll() is not None:
                self.root.after(0, self.on_emb_process_end)
                break
            time.sleep(1)

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
        # 停止硬件定时刷新
        if hasattr(self, 'hw_timer_id') and self.hw_timer_id:
            self.root.after_cancel(self.hw_timer_id)

        has_running = (
            (self.process and self.process.poll() is None) or
            (self.process_emb and self.process_emb.poll() is None)
        )
        if has_running:
            if messagebox.askyesno("确认退出", "当前有进程正在运行，确定要退出吗？"):
                self.stop_process()
                self.monitor_running = False
                self.monitor_emb_running = False
                self.root.destroy()
        else:
            self.monitor_running = False
            self.monitor_emb_running = False
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
