# 模型加载问题修复

## 问题现象

```
[Embedding] 离线加载失败，尝试在线模式: Cannot copy out of meta tensor; no data!
SearchEngine.query出错: Cannot copy out of meta tensor
警告：没有找到任何候选地块
```

## 根本原因

移动模型文件时，只复制了 `snapshots` 目录，但 **`blobs` 目录是空的**。

HuggingFace的缓存结构：
```
models--BAAI--bge-base-zh-v1.5/
├── blobs/              # 实际的模型权重文件（二进制blob）
│   ├── abc123...       # 真实的pytorch_model.bin
│   └── def456...       # 真实的其他文件
├── snapshots/          # 符号链接/引用
│   └── f03589ce.../    # 指向blobs中的文件
│       ├── pytorch_model.bin  → ../../blobs/abc123...
│       └── config.json        → ../../blobs/def456...
└── refs/
    └── main
```

**问题**：只复制 `snapshots` 会导致符号链接失效，模型无法加载。

---

## 解决方案

### 方案1：使用HuggingFace缓存（推荐，最简单）

**优点**：
- 无需复制文件
- 自动管理缓存
- 节省磁盘空间

**步骤**：

1. 修改 `config/app_config.json`：
```json
{
  "LOCAL_EMBEDDING_MODEL": "BAAI/bge-base-zh-v1.5"
}
```

2. 重启服务：
```bash
python server.py
```

**已完成** ✅

---

### 方案2：正确复制完整模型到项目目录

**优点**：
- 项目自包含
- 便于部署
- 不依赖用户缓存

**步骤**：

1. 运行修复脚本：
```bash
python scripts/fix_local_model.py
```

2. 脚本会：
   - 从 `~/.cache/huggingface/hub` 复制完整模型
   - 包括 `blobs` 和 `snapshots` 目录
   - 验证文件完整性

3. 更新配置（脚本会提示正确路径）

---

### 方案3：手动修复

如果自动脚本失败，手动操作：

```bash
# 1. 删除不完整的副本
Remove-Item -Recurse -Force ITINERA/model/llm_model/models--BAAI--bge-base-zh-v1.5

# 2. 完整复制
Copy-Item -Recurse `
  $env:USERPROFILE\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5 `
  ITINERA/model/llm_model/
```

---

## 验证修复

### 检查1：模型文件完整性

```bash
python scripts/find_model.py
```

应该看到：
```
✓ config.json (0.7 MB)
✓ pytorch_model.bin (400.0 MB)  # 关键！
✓ tokenizer_config.json (0.0 MB)
✓ modules.json (0.0 MB)
```

### 检查2：测试加载

```bash
python scripts/test_config.py
```

应该看到：
```
[Embedding] 加载本地模型 (离线模式)
[Embedding] ✓ 模型加载成功
✓ Embedding生成成功
  维度: 768
```

### 检查3：测试推荐

访问 http://localhost:8000，输入：
```
花都区食品生产厂用地
```

应该能看到推荐结果。

---

## 当前状态

✅ **已修复**：配置已更新为使用HuggingFace缓存

```json
{
  "LOCAL_EMBEDDING_MODEL": "BAAI/bge-base-zh-v1.5"
}
```

**下一步**：重启服务测试

```bash
python server.py
```

---

## 常见问题

### Q1: 为什么不能直接复制snapshots目录？

因为 `snapshots` 中的文件是符号链接或引用，指向 `blobs` 目录中的实际文件。只复制 `snapshots` 会导致链接失效。

### Q2: blobs目录里的文件名是什么？

是文件内容的SHA256哈希值，用于去重和完整性校验。

### Q3: 如何确认模型是否完整？

检查 `pytorch_model.bin` 文件大小：
- 应该约400MB
- 如果只有几KB，说明是符号链接，不是实际文件

### Q4: 使用HuggingFace缓存有什么缺点？

- 依赖用户缓存目录
- 部署时需要重新下载
- 多用户环境下每个用户都需要下载

### Q5: 如何在生产环境部署？

**选项A**：使用方案2，将完整模型打包到项目中

**选项B**：在部署脚本中自动下载：
```bash
python scripts/download_model.py
```

---

**修复时间**：2024-11-18 14:15  
**状态**：✅ 已修复（使用HuggingFace缓存）
