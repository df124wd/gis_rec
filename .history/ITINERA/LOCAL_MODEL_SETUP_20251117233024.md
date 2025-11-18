# 使用项目本地模型

## 优势

将模型放在项目目录下有以下好处：

1. **便于部署**：整个项目可以打包部署，无需重新下载模型
2. **版本控制**：模型版本固定，避免自动更新导致的问题
3. **离线运行**：完全不依赖网络和HuggingFace缓存
4. **团队协作**：团队成员共享同一模型版本

---

## 当前配置

### 模型位置
```
D:\gis_rec\ITINERA\model\llm_model\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65\
```

### 配置文件
`config/app_config.json`:
```json
{
  "EMBEDDING_PROVIDER": "local",
  "LOCAL_EMBEDDING_MODEL": "model/llm_model/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
}
```

---

## 目录结构

```
ITINERA/
├── model/
│   ├── llm_model/                    # 本地模型目录
│   │   └── models--BAAI--bge-base-zh-v1.5/
│   │       └── snapshots/
│   │           └── f03589ceff5aac7111bd60cfc7d497ca17ecac65/
│   │               ├── config.json
│   │               ├── pytorch_model.bin  (约400MB)
│   │               ├── tokenizer_config.json
│   │               ├── modules.json
│   │               └── ...
│   ├── data/                         # 数据集
│   ├── utils/                        # 工具类
│   └── ...
└── config/
    └── app_config.json               # 配置文件
```

---

## 验证配置

### 方法1：运行测试脚本
```powershell
python scripts/test_local_model.py
```

应该看到：
```
✓ 模型路径存在
✓ config.json (0.7 MB)
✓ pytorch_model.bin (400.0 MB)
✓ tokenizer_config.json (0.0 MB)
✓ modules.json (0.0 MB)
✓ 模型文件完整
✓ 模型加载成功
  维度: 768
✓ 所有测试通过！
```

### 方法2：直接启动服务
```powershell
python server.py
```

应该看到：
```
[Embedding] 使用项目本地模型: D:\gis_rec\ITINERA\model\llm_model\...
[Embedding] 加载本地模型 (离线模式)
[Embedding] ✓ 模型加载成功
```

---

## 部署到其他环境

### 1. 打包项目
```powershell
# 压缩整个项目目录
Compress-Archive -Path D:\gis_rec\ITINERA -DestinationPath ITINERA.zip
```

### 2. 在新环境解压
```bash
unzip ITINERA.zip
cd ITINERA
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 直接运行（无需下载模型）
```bash
python server.py
```

模型已包含在项目中，无需联网！

---

## 切换模型版本

### 使用不同的模型

如果想使用其他模型（如更小的模型），可以：

1. 下载新模型到项目目录：
```powershell
# 下载到临时位置
python scripts/download_model.py

# 移动到项目目录
Move-Item C:\Users\xiaoquanze\.cache\huggingface\hub\models--shibing624--text2vec-base-chinese `
          D:\gis_rec\ITINERA\model\llm_model\
```

2. 更新配置：
```json
{
  "LOCAL_EMBEDDING_MODEL": "model/llm_model/models--shibing624--text2vec-base-chinese/snapshots/xxx"
}
```

---

## 与缓存模式对比

| 特性 | 项目本地模型 | HuggingFace缓存 |
|------|-------------|----------------|
| **位置** | 项目目录 | `~/.cache/huggingface/` |
| **部署** | 打包即可 | 需重新下载 |
| **版本** | 固定 | 可能自动更新 |
| **网络** | 完全离线 | 首次需联网 |
| **磁盘** | 占用项目空间 | 占用用户缓存 |
| **共享** | 团队共享 | 每个用户独立 |

---

## 常见问题

### Q1: 模型文件太大，能压缩吗？
- 不建议压缩 `pytorch_model.bin`，会影响加载速度
- 可以使用Git LFS管理大文件
- 或使用更小的模型（如 `text2vec-base-chinese`，200MB）

### Q2: 如何更新模型？
```powershell
# 1. 下载新版本
python scripts/download_model.py

# 2. 替换旧模型
Remove-Item -Recurse D:\gis_rec\ITINERA\model\llm_model\models--BAAI--bge-base-zh-v1.5
Move-Item C:\Users\xiaoquanze\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5 `
          D:\gis_rec\ITINERA\model\llm_model\

# 3. 更新配置中的snapshot ID
```

### Q3: 能同时保留多个模型吗？
可以！只需在配置中切换：
```json
{
  "LOCAL_EMBEDDING_MODEL": "model/llm_model/models--BAAI--bge-base-zh-v1.5/snapshots/xxx",
  "_comment_alternative": "或使用: model/llm_model/models--shibing624--text2vec-base-chinese/snapshots/yyy"
}
```

### Q4: 如何备份模型？
```powershell
# 备份到外部存储
Copy-Item -Recurse D:\gis_rec\ITINERA\model\llm_model E:\backup\

# 或创建符号链接（节省空间）
New-Item -ItemType SymbolicLink -Path D:\gis_rec\ITINERA\model\llm_model `
         -Target E:\shared_models\bge-base-zh-v1.5
```

---

## Git管理建议

### .gitignore 配置

如果使用Git管理项目，建议：

**方案1：不提交模型（推荐）**
```gitignore
# .gitignore
model/llm_model/
```

团队成员各自下载模型到本地。

**方案2：使用Git LFS**
```bash
# 安装Git LFS
git lfs install

# 跟踪大文件
git lfs track "model/llm_model/**/*.bin"
git lfs track "model/llm_model/**/*.safetensors"

# 提交
git add .gitattributes
git add model/llm_model/
git commit -m "Add local model"
```

---

## 性能对比

| 场景 | 缓存模式 | 项目本地模式 |
|------|---------|-------------|
| **首次加载** | 2-3秒 | 2-3秒 |
| **后续加载** | 1-2秒 | 1-2秒 |
| **部署时间** | 5-10分钟（下载） | 0秒（已包含） |
| **磁盘占用** | 用户目录 | 项目目录 |

性能相同，但部署更快！

---

**更新时间**: 2024-11-17 23:30  
**状态**: ✅ 已配置项目本地模型
