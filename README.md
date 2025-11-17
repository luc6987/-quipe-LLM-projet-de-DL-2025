# Kaggle 2025 Competition: Influencers or Observers

本项目用于 Kaggle 竞赛 "Influencers or Observers: Predicting Social Roles" 的代码和实验。

## 📁 项目结构

```
Kaggle2025/
├── README.md                    # 项目说明文档（本文件）
├── initiate.py                  # 初始化脚本（下载数据、安装依赖）
├── environment.txt              # Python 依赖包列表
├── .gitignore                   # Git 忽略文件配置
│
├── data/                        # 数据目录（由 initiate.py 自动创建）
│   ├── train.jsonl             # 训练数据
│   └── kaggle_test.jsonl       # 测试数据
│
├── cache/                       # 缓存目录（中间处理结果）
│   ├── extracted_features.csv
│   ├── cleaned_features.csv
│   ├── test_extracted_features.csv
│   ├── test_cleaned_features.csv
│   ├── test_ids.csv
│   └── train_labels.csv
│
├── output/                      # 输出目录（预测结果）
│   └── xgboost_baseline.csv    # XGBoost 基线模型预测结果
│
└── src/                         # 源代码目录
    └── xgboost/                 # XGBoost 相关代码
        ├── xgboost_baseline.py  # XGBoost 基线模型主文件
        ├── train_xgboost.py    # 训练脚本
        ├── test_xgboost.py     # 测试脚本
        ├── extract_features.py # 特征提取
        ├── clean.py            # 数据清洗
        └── README_XGBOOST.md   # XGBoost 策略说明
```

## 🚀 快速开始

### 1. 配置 Kaggle API

在使用 `initiate.py` 下载数据之前，需要先配置 Kaggle API 凭证。

#### 步骤 1: 获取 API Token

1. 登录 [Kaggle](https://www.kaggle.com/)
2. 进入 [Account Settings](https://www.kaggle.com/account)
3. 在 "API" 部分，点击 "Create New Token"
4. 这会下载一个 `kaggle.json` 文件

#### 步骤 2: 放置 API Token

根据你的操作系统，将 `kaggle.json` 文件放在以下位置：

**macOS / Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json  # 设置正确的权限
```

**Windows:**
```powershell
# 创建目录（如果不存在）
mkdir %USERPROFILE%\.kaggle

# 移动文件（假设下载在 Downloads 文件夹）
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

#### 步骤 3: 验证配置

运行以下命令验证配置是否成功：

```bash
kaggle competitions list
```

如果能看到竞赛列表，说明配置成功！

### 2. 运行初始化脚本

`initiate.py` 脚本会自动完成以下任务：

1. ✅ **安装依赖包**：从 `environment.txt` 安装所有必需的 Python 包
2. ✅ **下载竞赛数据**：从 Kaggle 下载竞赛数据文件
3. ✅ **解压文件**：自动解压下载的 zip 文件
4. ✅ **整理文件结构**：将 `train.jsonl` 和 `kaggle_test.jsonl` 移动到 `data/` 目录
5. ✅ **清理临时文件**：删除多余的 zip 文件和其他临时文件

#### 使用方法

```bash
# 方法 1: 使用 Python 运行
python initiate.py

# 方法 2: 如果已设置可执行权限（macOS/Linux）
./initiate.py

# 方法 3: 使用 Python 3 明确指定
python3 initiate.py
```

#### 脚本输出示例

```
============================================================
Kaggle Competition Setup Script
============================================================

Step 1: Installing dependencies...
============================================================
Installing packages from environment.txt...
============================================================
Found 5 packages to install:
  - numpy
  - pandas
  - xgboost
  - scikit-learn
  - kaggle

Running: python -m pip install numpy pandas xgboost scikit-learn kaggle
Installation completed successfully!

Step 2: Downloading Kaggle data...
============================================================
Downloading Kaggle competition data...
============================================================
Running: kaggle competitions download -c influencers-or-observers-predicting-social-roles
Download completed!

Step 3: Extracting and organizing files...
============================================================
Extracting and organizing files...
============================================================
Created/verified data directory: /path/to/Kaggle2025/data
Extracting influencers-or-observers-predicting-social-roles.zip...
  Extracted influencers-or-observers-predicting-social-roles.zip
  Moved kaggle_test.jsonl to data/
  Moved train.jsonl to data/

Cleaning up extra files...
  Deleted influencers-or-observers-predicting-social-roles.zip

============================================================
Verification:
============================================================
✓ kaggle_test.jsonl found in data/
✓ train.jsonl found in data/

============================================================
Setup completed successfully!
============================================================
```

### 3. 运行模型

初始化完成后，可以运行模型训练和预测：

```bash
# 进入源代码目录
cd src/xgboost

# 运行 XGBoost 基线模型
python xgboost_baseline.py
```

## 📦 依赖包

项目所需的 Python 包列在 `environment.txt` 中：

- `numpy` - 数值计算
- `pandas` - 数据处理
- `xgboost` - 梯度提升决策树模型
- `scikit-learn` - 机器学习工具
- `kaggle` - Kaggle API 客户端

所有依赖包会在运行 `initiate.py` 时自动安装。

## 🔧 手动安装依赖（可选）

如果不想使用 `initiate.py`，也可以手动安装：

```bash
pip install -r environment.txt
```

或者：

```bash
pip install numpy pandas xgboost scikit-learn kaggle
```

## ⚠️ 常见问题

### 1. Kaggle API 认证失败

**错误信息：**
```
Error: Could not find kaggle.json. Make sure it's located in ~/.kaggle/
```

**解决方法：**
- 确保 `kaggle.json` 文件在正确的位置（见上面的配置步骤）
- 检查文件权限（macOS/Linux 需要设置为 600）
- 确认 JSON 文件格式正确

### 2. 下载失败

**错误信息：**
```
Error downloading data: 401 - Unauthorized
```

**解决方法：**
- 检查 Kaggle API token 是否过期（需要重新下载）
- 确认 Kaggle 账户已接受竞赛规则
- 验证网络连接

### 3. 权限错误（macOS/Linux）

**错误信息：**
```
Permission denied: ./initiate.py
```

**解决方法：**
```bash
chmod +x initiate.py
```

### 4. Python 版本问题

**要求：** Python 3.7 或更高版本

检查 Python 版本：
```bash
python --version
# 或
python3 --version
```

## 📝 注意事项

1. **数据文件**：`data/` 目录中的文件较大，已添加到 `.gitignore`，不会被 Git 跟踪
2. **缓存文件**：`cache/` 目录包含中间处理结果，可以删除后重新生成
3. **输出文件**：`output/` 目录包含模型预测结果，用于提交到 Kaggle

## 🔗 相关链接

- [Kaggle 竞赛页面](https://www.kaggle.com/competitions/influencers-or-observers-predicting-social-roles)
- [Kaggle API 文档](https://www.kaggle.com/docs/api)
- [XGBoost 文档](https://xgboost.readthedocs.io/)

## 📄 许可证

本项目仅用于学习和研究目的。

