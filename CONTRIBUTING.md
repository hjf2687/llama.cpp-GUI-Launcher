# 贡献指南

感谢你对本项目的关注！以下是参与贡献的方式。

## 🐛 报告 Bug

在提交 Issue 时，请包含以下信息：

1. **操作系统**：Windows 版本号
2. **Python 版本**：`python --version` 的输出
3. **llama.cpp 版本**：使用的编译版本
4. **复现步骤**：详细描述如何触发问题
5. **期望行为**：应该发生什么
6. **实际行为**：实际发生了什么
7. **日志输出**：启动器右侧日志窗口的错误信息

## 💡 功能建议

欢迎提出新功能建议！请在 Issue 中描述：

- 功能的具体用途
- 使用场景
- 期望的交互方式

## 🔧 提交代码

### 开发环境

1. 确保已安装 Python 3.10+
2. 克隆项目
3. 运行测试确认环境正常：
   ```bash
   python test_launcher.py
   ```

### 提交流程

1. Fork 本项目
2. 创建你的功能分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. 进行修改并确保代码风格一致
4. 运行测试：
   ```bash
   python test_launcher.py
   ```
5. 提交更改：
   ```bash
   git commit -m "feat: 添加某个功能"
   ```
6. 推送到你的 Fork：
   ```bash
   git push origin feature/your-feature-name
   ```
7. 在 GitHub 上创建 Pull Request

### 提交消息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

| 前缀 | 说明 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档更新 |
| `style:` | 代码格式（不影响逻辑） |
| `refactor:` | 重构（不新增功能/修复 Bug） |
| `test:` | 添加或修改测试 |
| `chore:` | 其他辅助修改 |

### 代码风格

- 使用 4 空格缩进
- 保持与现有代码风格一致
- 为新方法添加中文注释
- 复杂逻辑添加注释说明

## 📋 待办事项

如果你想贡献但不知道从哪里开始，可以关注这些方向：

- [ ] 添加 Linux/macOS 支持
- [ ] 添加模型下载功能
- [ ] 添加模型量化工具集成
- [ ] 添加更多语言的国际化支持
- [ ] 添加应用截图
- [ ] 添加单元测试

## ❓ 有任何问题？

如果你对贡献流程有任何疑问，欢迎通过 Issue 与我们沟通！
