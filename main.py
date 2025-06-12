import pandas as pd
import csv
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import time
from python_openai_messager.llm import send_llm_chat_request

# 全局锁，用于线程安全的文件写入
file_lock = Lock()

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
    # print(comment_text, 'start')
    try:
        # 发送请求给大模型
        response = send_llm_chat_request(
            prompt=prompt,
            stream=False      # 不需要流式输出
        )
        
        # 解析返回结果，提取 (需求, 情感) 元组
        result_tuple = parse_sentiment_result(response)
        # print(comment_text, 'done')
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

def load_processed_reply_ids(output_file_path):
    """
    加载已处理的reply_id，避免重复处理喵~
    """
    processed_ids = set()
    if os.path.exists(output_file_path):
        try:
            df = pd.read_csv(output_file_path, encoding='utf-8')
            if 'reply_id' in df.columns:
                processed_ids = set(df['reply_id'].astype(str))
            print(f"喵呜~发现已处理的评论 {len(processed_ids)} 条！继续上次的工作 ฅ^•ﻌ•^ฅ")
        except Exception as e:
            print(f"咪啾~读取历史结果文件出错: {str(e)} (´･ω･`)")
    return processed_ids

def save_batch_results_to_csv(results_batch, output_file_path):
    """
    批量保存结果到CSV文件（追加模式）喵~
    """
    if not results_batch:
        return
    
    with file_lock:
        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.exists(output_file_path)
        
        try:
            with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['reply_id', 'content', 'requirement', 'sentiment', 'llm_raw_response']
                deletions = []
                for result in results_batch:
                    if result['requirement'] is None and result['sentiment'] is None:
                        deletions.append(result)
                        continue
                    result['requirement'] = str(result['requirement']).replace('\n', ' ').replace('\r', ' ')
                    result['sentiment'] = str(result['sentiment']).replace('\n', ' ').replace('\r', ' ')
                    result['llm_raw_response'] = str(result['llm_raw_response']).replace('\n', ' ').replace('\r', ' ')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                results_batch = [result for result in results_batch if result not in deletions]
                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()
                
                # 写入数据
                for result in results_batch:
                    writer.writerow(result)
                    
        except Exception as e:
            print(f"咪啾~保存CSV文件失败: {str(e)} (´･ω･`)")

def process_comment_batch(batch_data, prompt_template, output_file_path, bar=None):
    """
    处理一批评论数据喵~
    """
    results_batch = []
    
    for _, row in batch_data.iterrows():
        reply_id = str(row.get('reply_id', ''))
        comment_content = str(row.get('content', '')).strip()
        
        if not comment_content or comment_content == 'nan':
            continue
        
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
            'reply_id': reply_id,
            'content': comment_content,
            'requirement': parsed_tuple[0] if parsed_tuple else None,
            'sentiment': parsed_tuple[1] if parsed_tuple else None,
            'llm_raw_response': raw_response
        }
        results_batch.append(result)
    
    # 批量保存到CSV
    save_batch_results_to_csv(results_batch, output_file_path)
    # 更新进度条
    if bar and len(results_batch) == 1:
        bar.write(f"==============================")
        bar.write(f"\t评论内容: {comment_content}")
    
    return len(results_batch)

def batch_analyze_comments_threaded(csv_file_path, output_file_path=None, length=-1, batch_size=1, max_workers=25, cooldown=2):
    """
    多线程批量分析评论数据喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - length: 处理的评论数量限制
    - batch_size: 每批处理的评论数量
    - max_workers: 最大工作线程数（根据每分钟2000的限制设置）
    """
    print("喵呜~多线程评论分析系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 加载提示词模板
    prompt_template = load_sentiment_prompt()
    prompt_template = prompt_template.replace('{product_name}', '特斯拉Model3')
    if not prompt_template:
        return []
    
    # 设置输出文件路径
    if not output_file_path:
        output_file_path = 'results/sentiment_analysis_results.csv'
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 加载已处理的reply_id
    processed_ids = load_processed_reply_ids(output_file_path)
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条评论数据喵~ (≧∇≦)ﾉ")
        
        # 过滤已处理的数据
        df = df[~df['reply_id'].astype(str).isin(processed_ids)]
        print(f"过滤后待处理 {len(df)} 条评论喵~ (≧▽≦)")
        
        if length > 0:
            df = df.head(length)
            print(f"将分析前 {length} 条评论喵~ (≧▽≦)")
            
    except Exception as e:
        print(f"咪啾~读取CSV文件失败: {str(e)} (´･ω･`)")
        return []
    
    if df.empty:
        print("咪啾~没有新的评论需要处理呢！(´･ω･`)")
        return []
    
    # 分批处理
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    batches = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batches.append(batch)
    
    print(f"将使用 {max_workers} 个线程处理 {total_batches} 个批次喵~ ✧(≖ ◡ ≖✿)")
    
    # 多线程处理
    processed_count = 0

    bar = tqdm(total=len(df), desc="处理评论中", unit="条")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_batch = {}
        for batch in batches:
            future = executor.submit(process_comment_batch, batch, prompt_template, output_file_path, bar)
            future_to_batch[future] = batch
        
        # 使用tqdm显示进度
        with bar as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_processed_count = future.result()
                    processed_count += batch_processed_count
                    pbar.update(batch_processed_count)
                    
                    # 简单的速率控制，避免超过API限制
                    time.sleep(cooldown)
                    
                except Exception as e:
                    print(f"咪啾~处理批次时出错: {str(e)} (´･ω･`)")
    
    print(f"批量分析完成！共处理 {processed_count} 条评论喵~ (ΦωΦ)")
    print(f"结果已保存到 {output_file_path} 喵~ ✧(≖ ◡ ≖✿)")
    
    return processed_count

def demo_single_analysis():
    """
    演示单条评论分析喵~
    """
    print("=== 单条评论分析演示 ===")
    
    prompt_template = load_sentiment_prompt()
    if not prompt_template:
        return
    
    prompt_template = prompt_template.replace('{product_name}', '特斯拉Model3')
    
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
    print("喵呜~多线程汽车评论情感分析系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 演示单条分析
    # demo_single_analysis()
    
    # 多线程批量分析评论
    processed_count = batch_analyze_comments_threaded(
        csv_file_path='data/comment_contents_cleaned.csv',
        output_file_path='results/sentiment_analysis_results.csv',
        length=100,  # 测试用，处理100条
        batch_size=1,  # 每批n条
        max_workers=30,  # 最大n个线程，控制并发
        cooldown = 1  # 每条评论处理后等待0.1秒，确保 max_workers * 60 <= cooldown * 2000
    )
    
    print(f"\n任务完成喵~处理了 {processed_count} 条评论！尾巴高速摇摆中！ヽ(=^･ω･^=)丿")
