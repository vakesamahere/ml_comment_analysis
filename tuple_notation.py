import pandas as pd
import csv
import re
import os
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from python_openai_messager.llm import send_llm_chat_request

# 全局变量
import aiofiles
import time
from datetime import timedelta

# 初始化全局计数器
failure_count = 0   # 失败次数
total_processed = 0 # 已处理总数
total_time = 0     # 总处理时间

def load_sentiment_prompt(prompt_file='prompts/report_sentiment_tuple.txt'):
    """
    加载情感分析提示词模板喵~
    
    参数:
    - prompt_file: 提示词文件路径
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        print(f"咪啾~找不到提示词文件呢！{prompt_file} (´･ω･`)")
        return None

async def analyze_comment_sentiment(comment_text, prompt_template):
    """
    分析单条评论的情感倾向和需求喵~（异步版本）
    
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
        response = await send_llm_chat_request(
            prompt=prompt,
            stream=False      # 不需要流式输出
        )
        if response.startswith("[error]"):
            raise ValueError(f"大模型请求失败: {response}")
    except Exception as e:
        print(f"咪啾~分析评论时出错啦: {str(e)} (´･ω･`)")
        return None
    
    try:
        # 解析返回结果，提取 (需求, 情感) 元组
        result_tuple = parse_sentiment_result(response)
        
        # 返回包含原始响应和解析结果的字典
        return {
            "parsed_result": result_tuple,
            "raw_response": response
        }
        
    except Exception as e:
        # print(f"咪啾~分析评论时出错啦: {str(e)} (´･ω･`)")
        return {
            "parsed_result": ('无关', '中性'),
            "raw_response": response
        }

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
    
    # print(f"咪啾~无法解析大模型返回结果: {response_text} (｡•́︿•̀｡)")
    return ('无关', '中性')  # 默认返回无关和中性

def load_processed_ids(output_file_path, id_column='reply_id'):
    """
    加载已处理的ID，避免重复处理喵~
    
    参数:
    - output_file_path: 输出文件路径
    - id_column: ID列的列名
    """
    processed_ids = set()
    if os.path.exists(output_file_path):
        try:
            df = pd.read_csv(output_file_path, encoding='utf-8')
            if id_column in df.columns:
                processed_ids = set(df[id_column].astype(str))
            print(f"喵呜~发现已处理的评论 {len(processed_ids)} 条！继续上次的工作 ฅ^•ﻌ•^ฅ")
        except Exception as e:
            print(f"咪啾~读取历史结果文件出错: {str(e)} (´･ω･`)")
    return processed_ids

async def save_batch_results_to_csv(results_batch, output_file_path, output_columns, pbar=None):
    """
    批量保存结果到CSV文件（追加模式）喵~（异步版本）
    
    参数:
    - results_batch: 要保存的批次结果
    - output_file_path: 输出文件路径
    - output_columns: 输出的列名列表
    - pbar: 进度条对象
    """
    global failure_count
    
    if not results_batch:
        return
    
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.exists(output_file_path)
    
    try:
        # 处理要写入的数据
        deletions = []
        for result in results_batch:
            if result.get('requirement') is None and result.get('sentiment') is None:
                deletions.append(result)
                failure_count += 1
                if pbar:
                    pbar.set_postfix({'失败': failure_count}, refresh=True)
                continue
                
            # 清理数据中的换行符
            for key in result:
                if isinstance(result[key], str):
                    result[key] = result[key].replace('\n', ' ').replace('\r', ' ')
        
        results_batch = [result for result in results_batch if result not in deletions]
        
        # 使用标准csv模块写入
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output_columns)
            
            if not file_exists:
                writer.writeheader()
            
            for result in results_batch:
                writer.writerow(result)
                
    except Exception as e:
        print(f"咪啾~保存CSV文件失败: {str(e)} (´･ω･`)")

async def process_comment_batch(batch_data, prompt_template, output_file_path, pbar=None, 
                              id_column='reply_id', content_column='content', 
                              output_columns=None):
    """
    处理一批评论数据喵~（异步版本）
    
    参数:
    - batch_data: 批次数据
    - prompt_template: 提示词模板
    - output_file_path: 输出文件路径
    - pbar: 进度条对象
    - id_column: ID列的列名
    - content_column: 内容列的列名
    - output_columns: 输出的列名列表
    """
    results_batch = []
    
    if output_columns is None:
        output_columns = [id_column, content_column, 'requirement', 'sentiment', 'llm_raw_response']
    
    for _, row in batch_data.iterrows():
        # 获取ID，如果不存在则使用索引作为ID
        item_id = str(row.get(id_column, '')) if id_column in row else str(_)
        comment_content = str(row.get(content_column, '')).strip()
        
        if not comment_content or comment_content == 'nan':
            continue
        
        # 分析情感
        sentiment_result = await analyze_comment_sentiment(comment_content, prompt_template)
        
        # 提取解析结果和原始响应
        if sentiment_result:
            parsed_tuple = sentiment_result.get("parsed_result")
            raw_response = sentiment_result.get("raw_response", "")
        else:
            parsed_tuple = None
            raw_response = ""
        
        # 保存结果
        result = {
            id_column: item_id,
            content_column: comment_content,
            'requirement': parsed_tuple[0] if parsed_tuple else None,
            'sentiment': parsed_tuple[1] if parsed_tuple else None,
            'llm_raw_response': raw_response
        }
        
        # 添加原始数据中的其他列
        for col in row.index:
            if col not in [id_column, content_column] and col not in result:
                result[col] = row[col]
                
        results_batch.append(result)
    
    # 批量保存到CSV（传递进度条）
    await save_batch_results_to_csv(results_batch, output_file_path, output_columns, pbar)
    
    return len(results_batch)

async def batch_analyze_comments_async(csv_file_path, output_file_path=None, length=-1, batch_size=1, 
                                     max_concurrent=25, cooldown=2, product_name='产品',
                                     id_column='reply_id', content_column='content', 
                                     extra_columns=None, prompt_file=None):
    """
    异步批量分析评论数据喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - length: 处理的评论数量限制
    - batch_size: 每批处理的评论数量
    - max_concurrent: 最大并发数
    - cooldown: 每条处理后的冷却时间（秒）
    - product_name: 产品名称，用于替换提示词中的占位符
    - id_column: ID列的列名
    - content_column: 内容列的列名
    - extra_columns: 需要保留的额外列名列表
    - prompt_file: 自定义提示词文件路径
    """
    print("喵呜~异步评论分析系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 加载提示词模板
    prompt_template = load_sentiment_prompt(prompt_file) if prompt_file else load_sentiment_prompt()
    if not prompt_template:
        return []
    
    prompt_template = prompt_template.replace('{product_name}', product_name)
    
    # 设置输出文件路径
    if not output_file_path:
        output_file_path = 'results/sentiment_analysis_results.csv'
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 确定要保留的列
    if extra_columns is None:
        extra_columns = []
    
    output_columns = [id_column, content_column, 'requirement', 'sentiment', 'llm_raw_response'] + extra_columns
    
    # 加载已处理的ID
    processed_ids = load_processed_ids(output_file_path, id_column)
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条评论数据喵~ (≧∇≦)ﾉ")
        
        # 确保必要的列存在
        if content_column not in df.columns:
            print(f"咪啾~找不到内容列 '{content_column}'! (´･ω･`)")
            return []
            
        # 如果ID列不存在，创建一个索引作为ID列
        if id_column not in df.columns:
            print(f"找不到ID列 '{id_column}'，将使用索引作为ID喵~")
            df[id_column] = df.index.astype(str)

        df = df[[id_column, content_column] + extra_columns] if extra_columns else df[[id_column, content_column]]
        
        # 过滤已处理的数据
        if len(processed_ids) > 0:
            df = df[~df[id_column].astype(str).isin(processed_ids)]
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
    
    print(f"将使用最大并发数 {max_concurrent} 处理 {total_batches} 个批次喵~ ✧(≖ ◡ ≖✿)")
    
    # 使用信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrent)
    processed_count = 0
    
    # 创建进度条
    pbar = async_tqdm(total=len(df), desc="处理评论中", unit="条")
    
    async def process_with_semaphore(batch):
        global total_processed, total_time
        
        async with semaphore:
            batch_start_time = asyncio.get_event_loop().time()
            result = await process_comment_batch(
                batch, prompt_template, output_file_path, pbar, 
                id_column, content_column, output_columns
            )
            
            # 更新计时器
            total_time = time.time() - start_time  # 更新总时间
            total_processed += len(batch)
            
            # 计算平均速度（条/分钟）
            avg_speed = (total_processed / total_time) if total_time > 0 else 0
            
            # 计算预计剩余时间
            remaining_items = len(df) - total_processed
            eta = (remaining_items / avg_speed) if avg_speed > 0 else 0
            eta_str = str(timedelta(seconds=int(eta))) if eta > 0 else "计算中..."
            
            # 计算失败率
            failure_rate = (failure_count / total_processed * 100) if total_processed > 0 else 0
            
            # 更新进度条信息
            pbar.set_postfix({
                '平均速度': f'{avg_speed:.1f}条/秒',
                '剩余时间': eta_str,
                '失败率': f'{failure_rate:.1f}%',
                '失败数量': failure_count
            }, refresh=True)
            
            # 更新进度条
            pbar.update(len(batch))
            
            # 控制请求速率
            time_taken = asyncio.get_event_loop().time() - batch_start_time
            need_cooldown = max(0, cooldown - time_taken)
            await asyncio.sleep(need_cooldown)
            
            return result
    
    # 创建所有任务
    tasks = [process_with_semaphore(batch) for batch in batches]
    
    # 使用gather执行所有任务
    results = await asyncio.gather(*tasks)
    processed_count = sum(results)
    
    # 关闭进度条（不需要await）
    pbar.close()
    
    # 计算总体统计信息
    total_time_mins = total_time / 60
    avg_speed = (processed_count / total_time) * 60 if total_time > 0 else 0
    failure_rate = (failure_count / processed_count * 100) if processed_count > 0 else 0
    
    print(f"批量分析完成！共处理 {processed_count} 条评论喵~ (ΦωΦ)")
    print(f"总用时: {total_time_mins:.1f}分钟")
    print(f"平均速度: {avg_speed:.1f}条/分钟")
    print(f"失败数: {failure_count} 条 (失败率: {failure_rate:.1f}%) (｡•́︿•̀｡)")
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
    global start_time  # 声明要使用全局变量
    print("喵呜~异步汽车评论情感分析系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 记录启动时间
    start_time = time.time()
    
    # 自定义配置
    config = {
        'csv_file_path': 'case_study/mtr/data/MTR.csv',
        'output_file_path': 'case_study/mtr/results/MTR_demand_tuple.csv',
        'length': -1,  # -1表示处理所有评论
        'batch_size': 1,  # 每批x条
        'max_concurrent': 90,  # 最大并发数
        'cooldown': 1,  # 每条评论处理后等待z秒
        'product_name': '麦当劳',  # 服务名称
        'id_column': 'id',  # ID列的列名
        'content_column': 'Content',  # 内容列的列名
        'extra_columns': [],  # 需要保留的额外列
        'prompt_file': 'case_study/mtr/prompts/tuple_generation.txt'  # 自定义提示词文件，None表示使用默认
    }
    
    # 演示单条分析
    # demo_single_analysis()
    
    # 异步批量分析评论
    asyncio.run(batch_analyze_comments_async(**config))
    
    print("\n任务完成喵~尾巴高速摇摆中！ヽ(=^･ω･^=)丿")
