import pandas as pd
import json
import re
from python_openai_messager.llm import send_llm_chat_request

def load_sentiment_prompt():
    """
    加载情感分析提示词模板喵~
    """
    try:
        with open('prompts/report_sentiment_tuple.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        print("咪啾~找不到提示词文件呢！(´･ω･`)")
        return None

def analyze_comment_sentiment(comment_text, prompt_template):
    """
    分析单条评论的情感倾向和需求喵~
    
    参数:
    - comment_text: 用户评论文本
    - prompt_template: 提示词模板
    
    返回:
    - dict: {"parsed_result": (需求, 情感), "raw_response": "原始回复"} 或 None（如果解析失败）
    """
    if not comment_text or not prompt_template:
        return None
    
    # 替换模板中的占位符
    prompt = prompt_template.replace('{raw_review_text}', comment_text)
    
    try:
        # 发送请求给大模型
        response = send_llm_chat_request(
            prompt=prompt,
            stream=False      # 不需要流式输出
        )
        # print("请求提示词\n==============")
        # print(prompt)
        # print("==============")
        # print(response)
        
        # 解析返回结果，提取 (需求, 情感) 元组
        result_tuple = parse_sentiment_result(response)
        
        # 返回包含原始响应和解析结果的字典
        return {
            "parsed_result": result_tuple,
            "raw_response": response
        }
        
    except Exception as e:
        print(f"咪啾~分析评论时出错啦: {str(e)} (´･ω･`)")
        return None

def parse_sentiment_result(response_text):
    """
    解析大模型返回的结果，提取(需求, 情感)元组喵~
    """
    if not response_text:
        return None
    
    # 尝试匹配 ('需求', '情感') 格式
    pattern = r"\('([^']+)',\s*'([^']+)'\)"
    match = re.search(pattern, response_text)
    
    if match:
        requirement = match.group(1).strip()
        sentiment = match.group(2).strip()
        return (requirement, sentiment)
    
    # 如果第一种格式不匹配，尝试其他可能的格式
    pattern2 = r'\("([^"]+)",\s*"([^"]+)"\)'
    match2 = re.search(pattern2, response_text)
    
    if match2:
        requirement = match2.group(1).strip()
        sentiment = match2.group(2).strip()
        return (requirement, sentiment)
    
    print(f"咪啾~无法解析大模型返回结果: {response_text} (｡•́︿•̀｡)")
    return None

def batch_analyze_comments(csv_file_path, output_file_path=None,length=-1):
    """
    批量分析评论数据喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出结果文件路径（可选）
    
    返回:
    - list: 分析结果列表
    """
    print("喵呜~开始批量分析评论啦！ฅ^•ﻌ•^ฅ")
    
    # 加载提示词模板
    prompt_template = load_sentiment_prompt()
    prompt_template = prompt_template.replace('{product_name}', '特斯拉Model3')
    if not prompt_template:
        return []
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条评论数据喵~ (≧∇≦)ﾉ")
        if length > 0:
            df = df.head(length)
            print(f"将分析前 {length} 条评论喵~ (≧▽≦)")
    except Exception as e:
        print(f"咪啾~读取CSV文件失败: {str(e)} (´･ω･`)")
        return []
    
    results = []
    
    # 逐条分析评论
    for index, row in df.iterrows():
        comment_content = str(row.get('content', '')).strip()
        
        if not comment_content or comment_content == 'nan':
            continue
        
        print(f"分析第 {index + 1} 条评论中...喵~ ")
        
        # 分析情感
        sentiment_result = analyze_comment_sentiment(comment_content, prompt_template)
        
        # 提取解析结果和原始响应
        if sentiment_result:
            parsed_tuple = sentiment_result.get("parsed_result")
            raw_response = sentiment_result.get("raw_response", "")
        else:
            parsed_tuple = None
            raw_response = ""
        
        # 保存结果
        result = {
            'global_id': row.get('global_id', ''),
            'content': comment_content,
            'sentiment_analysis': parsed_tuple,
            'requirement': parsed_tuple[0] if parsed_tuple else None,
            'sentiment': parsed_tuple[1] if parsed_tuple else None,
            'llm_raw_response': raw_response  # 新增：保存LLM原始响应
        }
        results.append(result)
        
        # 简单的进度显示
        if (index + 1) % 10 == 0:
            print(f"已完成 {index + 1} 条评论分析喵~ ヽ(=^･ω･^=)丿")
    
    print(f"批量分析完成！共处理 {len(results)} 条评论喵~ (ΦωΦ)")
    
    # 保存结果到文件
    if output_file_path:
        save_results_to_file(results, output_file_path)
    
    return results

def save_results_to_file(results, output_file_path):
    """
    保存分析结果到文件喵~
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_file_path} 喵~ ✧(≖ ◡ ≖✿)")
    except Exception as e:
        print(f"咪啾~保存文件失败: {str(e)} (´･ω･`)")

def demo_single_analysis():
    """
    演示单条评论分析喵~
    """
    print("=== 单条评论分析演示 ===")
    
    prompt_template = load_sentiment_prompt()
    if not prompt_template:
        return
    
    # 示例评论
    test_comments = [
        "这辆车的加速性能真是太棒了，轻轻一点油门就能感受到推背感！",
        "导航系统经常卡顿，路线规划也不准确，让我很困扰。",
        "我觉得车内空间还可以，不算大但也不挤。",
        "健康度真不错，现在用LG的三元锂肯定不如之前的松下"
    ]
    
    for comment in test_comments:
        print(f"\n评论: {comment}")
        result = analyze_comment_sentiment(comment, prompt_template)
        if result:
            parsed_result = result.get("parsed_result")
            raw_response = result.get("raw_response", "")
            print(f"解析结果: {parsed_result}")
            print(f"LLM原始响应: {raw_response}")
        else:
            print("分析结果: None")
        print("-" * 50)

if __name__ == "__main__":
    print("喵呜~汽车评论情感分析系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 演示单条分析
    # demo_single_analysis()
    
    # 批量分析评论
    results = batch_analyze_comments(
        csv_file_path='data/comment_contents_cleaned.csv',
        output_file_path='results/sentiment_analysis_results.json',
        length = 5,
    )
    
    print("\n任务完成喵~尾巴高速摇摆中！ヽ(=^･ω･^=)丿")
