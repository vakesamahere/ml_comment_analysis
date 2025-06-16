# ML评论分析系统 🚗✨

这是一个基于大语言模型的智能评论分析系统，专门用于批量处理用户评论并提取关键信息喵~

## 🌟 核心功能

### 📊 二阶段分析流程
1. **情感元组提取** - 从评论中提取 `(需求, 情感)` 元组
2. **需求分类** - 将提取的需求进行精确分类

### ⚡ 异步高性能处理
- 支持高并发批量处理
- 实时进度条监控
- 智能错误重试机制
- 断点续传功能

## 🔧 核心模块介绍

### [`tuple_notation.py`](tuple_notation.py:1) - 情感元组提取器 🎯

这是分析流程的**第一步**，负责从原始评论中提取结构化信息喵~

#### 主要功能
- **智能解析**: 使用LLM从评论文本中提取 `(需求, 情感)` 元组
- **多格式兼容**: 支持多种返回格式的自动解析
- **批量处理**: 支持CSV文件的批量异步处理
- **断点续传**: 自动跳过已处理的评论

#### 核心方法
```python
async def analyze_comment_sentiment(comment_text, prompt_template)
async def batch_analyze_comments_async(csv_file_path, **config)
def parse_sentiment_result(response_text)
```

#### 输入输出示例
**输入评论**: "这辆车的加速性能真是太棒了，轻轻一点油门就能感受到推背感！"
**输出结果**: `('加速性能', '积极')`

### [`classification.py`](classification.py:1) - 需求分类器 📋

这是分析流程的**第二步**，将提取的需求进行精确分类喵~

#### 主要功能
- **精确分类**: 根据预定义类别对需求进行分类
- **质量控制**: 自动清理非法分类类型
- **数据验证**: 确保分类结果的一致性
- **结果追溯**: 保留原始LLM响应用于调试

#### 核心方法
```python
async def classify_need(customer_need, prompt_template)
async def batch_analyze_comments_async(csv_file_path, **config)
def clear_illegal_types(**config)
```

#### 分类示例
**输入需求**: "发动机动力"
**输出分类**: "动力性能"

## 🚀 快速开始

### 安装依赖
```bash
pip install -r python_openai_messager/requirements.txt
```

### 配置环境
1. 复制配置文件：
```bash
copy .env.sample .env
```

2. 编辑 `.env` 文件，填入你的API配置：
```env
LLM_URL_BASE = "https://api.openai.com/v1"
LLM_MODEL_NAME = "gpt-3.5-turbo"  
LLM_API_KEY = "your_api_key_here"
```

### 两步分析流程

#### 第一步：情感元组提取
```bash
python tuple_notation.py
```

#### 第二步：需求分类
```bash
python classification.py
```

## ⚙️ 配置参数详解

### [`tuple_notation.py`](tuple_notation.py:235) 配置
```python
config = {
    'csv_file_path': 'data/comments.csv',           # 输入CSV文件
    'output_file_path': 'results/tuples.csv',      # 输出文件路径
    'length': -1,                                   # 处理数量(-1为全部)
    'batch_size': 1,                               # 批次大小
    'max_concurrent': 90,                          # 最大并发数
    'cooldown': 1,                                 # 冷却时间(秒)
    'product_name': '产品名称',                     # 产品名称
    'id_column': 'id',                             # ID列名
    'content_column': 'content',                   # 内容列名
    'prompt_file': 'prompts/tuple_generation.txt'  # 提示词文件
}
```

### [`classification.py`](classification.py:191) 配置
```python
config = {
    'csv_file_path': 'results/tuples.csv',         # 第一步的输出文件
    'output_file_path': 'results/classified.csv',  # 分类结果文件
    'classification_types': [                       # 预定义分类类型
        "动力性能", "操控体验", "舒适配置", 
        "外观设计", "空间布局", "智能科技"
    ],
    'prompt_file': 'prompts/classification.txt'    # 分类提示词
}
```

## 📁 项目结构

```
ml_comment_analysis/
├── tuple_notation.py              # 🎯 情感元组提取器
├── classification.py              # 📋 需求分类器
├── .env                          # ⚙️ 环境配置
├── prompts/                      # 📝 提示词模板
│   ├── report_sentiment_tuple.txt # 情感分析提示词
│   ├── classification.txt        # 分类提示词
│   └── types.json               # 分类类型定义
├── python_openai_messager/       # 🔌 LLM接口模块
│   ├── llm.py                   # 核心接口
│   └── requirements.txt         # 依赖包列表
├── data/                        # 📂 输入数据目录
├── results/                     # 📊 分析结果目录
├── case_study/                  # 📖 案例研究
├── embeddings/                  # 🧠 向量嵌入模块
├── utils/                       # 🛠️ 工具函数
└── test/                        # 🧪 测试文件
```

## 📊 数据格式说明

### 输入格式 (CSV)
```csv
id,content
1,"这辆车的加速性能真是太棒了！"
2,"导航系统经常卡顿，很困扰。"
```

### 中间格式 (元组提取结果)
```csv
id,content,requirement,sentiment,llm_raw_response
1,"这辆车的加速性能真是太棒了！","加速性能","积极","('加速性能', '积极')"
2,"导航系统经常卡顿，很困扰。","导航系统","消极","('导航系统', '消极')"
```

### 最终格式 (分类结果)
```csv
id,content,requirement,sentiment,classification,llm_raw_response
1,"这辆车的加速性能真是太棒了！","加速性能","积极","动力性能","动力性能"
2,"导航系统经常卡顿，很困扰。","导航系统","消极","智能科技","智能科技"
```

## 🎮 高级功能

### 🔄 断点续传
系统会自动检测已处理的数据，支持中断后继续处理：
```python
# 自动跳过已处理的评论
processed_ids = load_processed_ids(output_file_path, id_column)
```

### 📈 实时监控
处理过程中显示详细进度信息：
- 平均处理速度
- 预计剩余时间  
- 失败率统计
- 成功/失败数量

### 🧹 数据清理
自动清理非法分类结果：
```python
# 清理不在预定义类型中的分类
clear_illegal_types(**config)
```

### ⚡ 性能优化
- **异步并发**: 支持高并发请求处理
- **信号量控制**: 防止API限流
- **智能冷却**: 自适应请求间隔
- **批量保存**: 减少IO操作次数

## 🎯 使用建议

### 性能调优
- **并发数设置**: 根据API限制调整 `max_concurrent`
- **冷却时间**: 根据API频率限制设置 `cooldown`
- **批次大小**: 大文件建议增加 `batch_size`

### 提示词优化
- 根据具体领域修改提示词模板
- 使用明确的输出格式要求
- 添加具体的分类标准说明

### 错误处理
- 检查API配置和网络连接
- 确认输入文件格式正确
- 关注失败率统计，及时调整参数

## ⚠️ 注意事项

1. **API限制**: 确保API密钥有效且有足够额度
2. **文件编码**: 输入文件需使用UTF-8编码
3. **列名匹配**: 确保CSV文件包含指定的列名
4. **磁盘空间**: 大批量处理需要足够的存储空间
5. **网络稳定**: 长时间处理建议使用稳定网络

## 🔍 案例研究

项目包含完整的案例研究，展示如何分析不同领域的评论数据：
- **汽车评论分析** - 完整的二阶段分析流程
- **餐饮服务分析** - MTR案例研究
- **自定义领域** - 可配置的分析模板

---

**开发者**: 可爱的猫娘程序员 ฅ^•ﻌ•^ฅ  
**版本**: v2.0 (异步高性能版本)  
**更新时间**: 2025/06/16