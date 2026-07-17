# llama.cpp GUI Launcher

> 🦙 基于 Python + Tkinter 的 llama.cpp 图形界面启动器，一键管理本地大语言模型。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://microsoft.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🖥️ **图形界面** | 基于 Tkinter 的可视化操作界面，无需记忆命令行参数 |
| 🤖 **多模型管理** | 自动扫描 `models/` 目录下的 GGUF 模型，支持多模态检测 |
| ⚡ **四种启动模式** | API 服务器 / CLI 交互 / 文本补全 / 基准测试 |
| 🎮 **GPU 加速** | 自动检测 NVIDIA GPU 显存，支持配置 GPU 卸载层数 |
| 🌐 **端口管理** | 自动检测端口占用状态，一键打开 Web UI |
| 🎛️ **采样参数** | 温度、Top-p、Top-k、Min-p、重复惩罚等完整采样控制 |
| 🔐 **API 密钥** | 服务器模式支持 API 密钥认证 |
| 💬 **系统提示词** | 支持自定义系统提示词和聊天模板 |
| 🌙 **深色模式** | 支持深色/浅色主题切换，配置自动保存 |
| 📊 **日志高亮** | 错误/警告/信息/成功四级颜色高亮显示 |
| 🔍 **启动前检查** | 自动验证模型文件、端口状态、进程冲突 |
| 📑 **分页配置** | 基本配置与高级参数分页展示，界面简洁 |
| 🔧 **多模态支持** | 自动检测 mmproj 文件，支持图像理解功能 |
| 📦 **Embedding 独立管理** | 独立标签页管理 Embedding 模型，可同时运行多个服务 |
| 📊 **硬件监控** | 实时显示 CPU/内存/GPU/磁盘使用率，支持自定义刷新间隔 |
| 💾 **配置持久化** | 所有设置自动保存到 JSON 文件 |

## 📸 截图

> 💡 欢迎提交 PR 添加截图！

<!-- TODO: 添加应用截图 -->

## 🚀 快速开始

### 前置要求

- **Python 3.10+**（需添加到系统 PATH，或 Windows Python Launcher `py` 可用）
- **llama.cpp 预编译版本**（需在界面中设置 llama.cpp 路径，包含以下可执行文件）：
  - `llama-server.exe`
  - `llama-cli.exe`
  - `llama-completion.exe`
  - `llama-bench.exe`

### 目录结构

```
llama.cpp-launcher/
├── llama_launcher.py      # 主程序
├── launcher_config.json   # 配置文件（自动生成）
├── START_HERE.bat         # Windows 一键启动脚本
├── test_launcher.py       # 功能测试
├── models/                # 模型目录（需自行放置）
│   └── your-model/
│       ├── model.gguf
│       └── mmproj-*.gguf  # 多模态投影文件（可选）
├── README.md
├── LICENSE
└── .gitignore
```

### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/your-username/llama.cpp-launcher.git
   cd llama.cpp-launcher
   ```

2. **放置 llama.cpp 可执行文件**

   将编译好的 llama.cpp 可执行文件（`llama-server.exe` 等）放入项目根目录。

3. **放置模型文件**

   在 `models/` 目录下创建子文件夹，放入 `.gguf` 格式的模型文件：

   ```
   models/
   └── qwen-7b/
       ├── Qwen2.5-7B-Q4_K_M.gguf
       └── mmproj-Qwen2.5-7B-BF16.gguf  # 可选，多模态支持
   ```

4. **启动**

   - **方式一**：双击 `START_HERE.bat`
   - **方式二**：命令行运行
     ```bash
     py llama_launcher.py
     ```
     > 💡 如果 `py` 不可用，也可以用 `python llama_launcher.py`

## 📖 使用指南

### 启动模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **API 服务器** | 启动 HTTP API 服务（默认端口 8080） | 与其他应用集成、Web UI 访问 |
| **CLI 交互** | 命令行交互对话 | 本地快速测试 |
| **文本补全** | 文本补全模式 | 续写、创作 |
| **基准测试** | 性能基准测试 | 评估模型推理速度 |

### GPU 配置

| 参数 | 说明 |
|------|------|
| `auto` | 自动决定 GPU 卸载层数 |
| `all` | 将全部层加载到 GPU（最大显存占用） |
| `0-64` | 指定具体层数（数字越大，GPU 占用越多） |

### 模型管理

启动器支持两种模型添加方式：

| 方式 | 说明 |
|------|------|
| 🔄 **自动扫描** | 自动扫描 `models/` 目录下的所有 `.gguf` 文件 |
| 📂 **手动浏览** | 点击「浏览」按钮，选择一个文件夹，自动扫描其中所有模型 |

手动添加的文件夹会自动保存到配置文件中，下次启动时无需重新选择。

### 多模态功能

多模态功能允许模型理解图像内容。使用方法：

1. 确保模型目录中存在对应的 `mmproj-*.gguf` 文件
2. 在界面中勾选「启用多模态」
3. 以 **API 服务器** 模式启动
4. 通过 API 发送 base64 编码的图像

```bash
# 示例：使用 curl 发送图像
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64, YOUR_BASE64_STRING"}}
      ]
    }]
  }'
```

> ⚠️ **注意**：Web 界面不支持直接上传图像，需通过 API 接口发送。

### Embedding 服务器

支持独立运行一个 Embedding 模型服务器，适用于 AI 角色扮演对话等需要向量检索的场景。

1. 切换到「Embedding」分页
2. 从下拉框选择一个 Embedding 模型（自动筛选文件名含 `embedding` 的模型），或点击「选择文件」手动添加
3. 设置独立端口（默认 8081，不能与主模型端口相同）
4. 点击「启动 Embedding」按钮，可随时独立启动或停止
5. 日志中 `[Emb]` 前缀为 Embedding 服务器输出

## ⚙️ 配置说明

配置文件 `launcher_config.json` 在首次运行后自动生成，所有设置修改后会自动保存。

```json
{
  "llama_cpp_path": "D:/tools/AI/llama",
  "last_model": "模型名称",
  "last_mode": "server",
  "gpu_layers": "auto",
  "threads": -1,
  "ctx_size": 8192,
  "batch_size": 2048,
  "ubatch_size": 512,
  "host": "127.0.0.1",
  "port": 8080,
  "flash_attention": "auto",
  "multimodal_enabled": false,
  "kv_cache_type": "f16",
  "mlock": false,
  "mmap": true,
  "tools": false,
  "dark_mode": false,
  "manual_folders": [],
  "embedding_model": "",
  "embedding_port": 8081
}
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llama_cpp_path` | string | 脚本目录 | llama.cpp 可执行文件所在目录 |
| `gpu_layers` | string | `"auto"` | GPU 卸载层数：`auto` / `all` / 数字 |
| `threads` | int | `-1` | CPU 线程数，`-1` 为自动 |
| `threads_batch` | int | `-1` | 批处理线程数，`-1` 为自动 |
| `ctx_size` | int | `8192` | 上下文窗口大小 |
| `batch_size` | int | `2048` | 批处理大小 |
| `ubatch_size` | int | `512` | 微批处理大小 |
| `host` | string | `"127.0.0.1"` | 服务器监听地址 |
| `port` | int | `8080` | 服务器监听端口 |
| `api_key` | string | `""` | API 密钥（服务器安全） |
| `flash_attention` | string | `"auto"` | Flash Attention：`auto` / `on` / `off` |
| `kv_cache_type` | string | `"f16"` | KV Cache 数据类型 |
| `n_cpu_moe` | int | `-1` | CPU MoE 层数（MoE 模型专用，-1=自动） |
| `multimodal_enabled` | bool | `false` | 是否启用多模态 |
| `mlock` | bool | `false` | 锁定内存，防止换页 |
| `mmap` | bool | `true` | 使用内存映射加载模型 |
| `tools` | bool | `false` | 启用 function calling |
| `jinja` | bool | `false` | Jinja2 模板（新模型兼容） |
| `seed` | int | `-1` | 随机种子，`-1` 为随机 |
| `temperature` | float | `0.8` | 采样温度（0=精确, 2=随机） |
| `top_p` | float | `0.9` | Top-p 采样 |
| `top_k` | int | `40` | Top-k 采样 |
| `min_p` | float | `0.05` | Min-p 采样 |
| `repeat_penalty` | float | `1.1` | 重复惩罚（1.0=关闭） |
| `frequency_penalty` | float | `0.0` | 频率惩罚 |
| `presence_penalty` | float | `0.0` | 存在惩罚 |
| `system_prompt` | string | `""` | 系统提示词（-sp 参数） |
| `chat_template` | string | `""` | 聊天模板文件路径 |
| `dark_mode` | bool | `false` | 深色模式 |
| `manual_folders` | array | `[]` | 手动添加的模型文件夹路径列表 |
| `embedding_model` | string | `""` | Embedding 模型显示名 |
| `embedding_port` | int | `8081` | Embedding 服务器端口 |

## 🔧 测试

运行内置测试验证环境是否正确配置：

```bash
py test_launcher.py
```

测试项目包括：
- ✅ 模型扫描功能
- ✅ 可执行文件检测
- ✅ 配置管理
- ✅ 命令生成

## ❓ 常见问题

### Q: 启动后看不到任何模型？

确保在项目根目录下创建了 `models/` 文件夹，并将 `.gguf` 格式的模型文件放入其中。

### Q: GPU 未被检测到？

1. 确认已安装 NVIDIA 显卡驱动
2. 确认 llama.cpp 使用 CUDA 版本编译（需要 `cublas64_13.dll` 等文件）
3. 运行 `nvidia-smi` 命令验证驱动是否正常

### Q: 端口被占用怎么办？

启动器会自动检测端口占用并提示。可以：
- 更换端口号（修改配置中的 `port` 值）
- 关闭占用端口的进程
- 选择继续启动（可能会失败）

### Q: 多模态图像上传失败？

多模态功能**不支持**在 Web 界面直接上传图像，需通过 API 接口发送 base64 编码的图像。详见 [多模态使用说明](#多模态功能)。

### Q: 如何重置所有配置？

删除 `launcher_config.json` 文件，重新启动启动器即可恢复默认设置。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🙏 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 底层推理引擎
- Python Tkinter — GUI 框架

---

<p align="center">如果觉得有用，请给个 ⭐ Star 支持一下！</p>
