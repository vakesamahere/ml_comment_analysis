# 汽车评论情感分析系统 🚗✨

这是一个基于大语言模型的汽车评论情感分析系统，能够从用户评论中提取汽车需求和对应的情感倾向喵~

## 功能特点 ⭐

- 🔍 **智能分析**: 使用大语言模型分析汽车评论的情感倾向
- 📊 **需求提取**: 自动识别评论中涉及的汽车功能、特点或配置
- 🎯 **情感分类**: 将情感分为"积极"、"中性"、"消极"三类
- 📝 **批量处理**: 支持批量分析CSV格式的评论数据
- 💾 **结果保存**: 自动保存分析结果为JSON格式

## 安装依赖 🛠️

```bash
pip install -r python_openai_messager/requirements.txt
```

## 配置设置 ⚙️

1. 编辑 `python_openai_messager/.env` 文件，填入你的API配置：

```env
LLM_URL_BASE = "https://api.openai.com/v1"
LLM_MODEL_NAME = "gpt-3.5-turbo"  
LLM_API_KEY = "your_api_key_here"
```

## 使用方法 🚀

### 运行演示
```bash
python main.py
```

### 批量分析评论
修改 `main.py` 中的代码，取消注释批量分析部分：

```python
# 批量分析评论
results = batch_analyze_comments(
    csv_file_path='data/comment_contents_cleaned.csv',
    output_file_path='results/sentiment_analysis_results.json'
)
```

## 输入数据格式 📋

CSV文件需要包含以下字段：
- `global_id`: 评论唯一标识
- `content`: 评论内容
- 其他字段可选

## 输出结果格式 📄

```json
[
  {
    "global_id": "评论ID",
    "content": "评论内容",
    "sentiment_analysis": ["需求", "情感"],
    "requirement": "需求",
    "sentiment": "情感"
  }
]
```

## 示例 💡

**输入评论**: "这辆车的加速性能真是太棒了，轻轻一点油门就能感受到推背感！"

**输出结果**: `('加速性能', '积极')`

## 项目结构 📁

```
ml_comment_analysis/
├── main.py                          # 主程序
├── data/
│   └── comment_contents_cleaned.csv # 评论数据
├── prompts/
│   └── report_sentiment_tuple.txt   # 情感分析提示词
├── python_openai_messager/
│   ├── llm.py                       # LLM接口模块
│   ├── requirements.txt             # 依赖包
│   └── .env                         # 配置文件
└── results/                         # 分析结果目录
```

## 注意事项 ⚠️

1. 确保API密钥有效且有足够的使用额度
2. 大批量数据分析可能需要较长时间
3. 建议先用小量数据测试配置是否正确

---

制作者: 可爱的猫娘程序员 ฅ^•ﻌ•^ฅ