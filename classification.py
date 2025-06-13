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

def load_classification_prompt():
    """
    加载分类提示词模板喵~
    """
    try:
        with open('prompts/classification.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        print("咪啾~找不到提示词文件呢！(´･ω･`)")
        return None

async def classify_need(customer_need, prompt_template):
    """
    分析单个需求的分类喵~（异步版本）
    
    参数:
    - customer_need: 用户需求文本
    - prompt_template: 提示词模板
    
    返回:
    - dict: {"parsed_result": "分类结果", "raw_response": "原始回复"} 或 None（如果解析失败）
    """
    if not customer_need or not prompt_template:
        return None
    
    # 替换模板中的占位符
    prompt = prompt_template.replace('{customer_need}', customer_need)
    
    try:
        # 发送请求给大模型
        response = await send_llm_chat_request(
            prompt=prompt,
            stream=False      # 不需要流式输出
        )
        if response.startswith("[error]"):
            raise ValueError(f"大模型请求失败: {response}")
    except Exception as e:
        print(f"咪啾~分析需求时出错啦: {str(e)} (´･ω･`)")
        return None
    
    try:
        # 解析返回结果
        result = response.strip()
        
        # 返回包含原始响应和解析结果的字典
        return {
            "parsed_result": result,
            "raw_response": response
        }
        
    except Exception as e:
        return {
            "parsed_result": "其他",
            "raw_response": response
        }

async def save_batch_results_to_csv(results_batch, output_file_path, pbar=None):
    """
    批量保存结果到CSV文件（追加模式）喵~（异步版本）
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
            if result['classification'] is None:
                deletions.append(result)
                failure_count += 1
                if pbar:
                    pbar.set_postfix({'失败': failure_count}, refresh=True)
                continue
            result['classification'] = str(result['classification']).replace('\n', ' ').replace('\r', ' ')
            result['llm_raw_response'] = str(result['llm_raw_response']).replace('\n', ' ').replace('\r', ' ')
        
        results_batch = [result for result in results_batch if result not in deletions]
        
        # 使用标准csv模块写入
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['reply_id', 'content', 'requirement', 'sentiment', 'classification', 'llm_raw_response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for result in results_batch:
                writer.writerow(result)
                
    except Exception as e:
        print(f"咪啾~保存CSV文件失败: {str(e)} (´･ω･`)")

async def process_comment_batch(batch_data, prompt_template, output_file_path, pbar=None):
    """
    处理一批评论数据喵~（异步版本）
    """
    results_batch = []
    
    for _, row in batch_data.iterrows():
        reply_id = str(row.get('reply_id', ''))
        content = str(row.get('content', '')).strip()
        requirement = str(row.get('requirement', '')).strip()
        sentiment = str(row.get('sentiment', '')).strip()
        
        if not requirement or requirement == 'nan':
            continue
        
        # 分类分析
        classification_result = await classify_need(requirement, prompt_template)
        
        # 提取解析结果和原始响应
        if classification_result:
            parsed_result = classification_result.get("parsed_result")
            raw_response = classification_result.get("raw_response", "")
        else:
            parsed_result = None
            raw_response = ""
        
        # 保存结果
        result = {
            'reply_id': reply_id,
            'content': content,
            'requirement': requirement,
            'sentiment': sentiment,
            'classification': parsed_result,
            'llm_raw_response': raw_response
        }
        results_batch.append(result)
    
    # 批量保存到CSV（传递进度条）
    await save_batch_results_to_csv(results_batch, output_file_path, pbar)
    
    return len(results_batch)

async def batch_analyze_comments_async(csv_file_path, output_file_path=None, length=-1, batch_size=1, max_concurrent=25, cooldown=2):
    """
    异步批量分析评论数据喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - length: 处理的评论数量限制
    - batch_size: 每批处理的评论数量
    - max_concurrent: 最大并发数
    """
    print("喵呜~异步评论分类系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 加载提示词模板
    prompt_template = load_classification_prompt()
    if not prompt_template:
        return []
    
    # 设置输出文件路径
    if not output_file_path:
        output_file_path = 'results/classification_results.csv'
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条数据喵~ (≧∇≦)ﾉ")
        
        processed_ids = pd.read_csv(output_file_path, encoding='utf-8')['reply_id'].astype(str).tolist() if os.path.exists(output_file_path) else []
        # 过滤已处理的数据
        df = df[~df['reply_id'].astype(str).isin(processed_ids)]
        print(f"过滤后待处理 {len(df)} 条评论喵~ (≧▽≦)")
        
        if length > 0:
            df = df.head(length)
            print(f"将分析前 {length} 条数据喵~ (≧▽≦)")
            
    except Exception as e:
        print(f"咪啾~读取CSV文件失败: {str(e)} (´･ω･`)")
        return []
    
    if df.empty:
        print("咪啾~没有数据需要处理呢！(´･ω･`)")
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
            result = await process_comment_batch(batch, prompt_template, output_file_path, pbar)
            
            # 计算处理时间（不包含cooldown）
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
    
    # 关闭进度条
    pbar.close()
    
    # 计算总体统计信息
    total_time_mins = total_time / 60
    avg_speed = (processed_count / total_time) * 60 if total_time > 0 else 0
    failure_rate = (failure_count / processed_count * 100) if processed_count > 0 else 0
    
    print(f"批量分析完成！共处理 {processed_count} 条数据喵~ (ΦωΦ)")
    print(f"总用时: {total_time_mins:.1f}分钟")
    print(f"平均速度: {avg_speed:.1f}条/分钟")
    print(f"失败数: {failure_count} 条 (失败率: {failure_rate:.1f}%) (｡•́︿•̀｡)")
    print(f"结果已保存到 {output_file_path} 喵~ ✧(≖ ◡ ≖✿)")
    
    return processed_count

def demo_single_analysis():
    """
    演示单条需求分类喵~
    """
    print("=== 单条需求分类演示 ===")
    
    prompt_template = load_classification_prompt()
    if not prompt_template:
        return
    
    # 示例需求
    test_needs = [
        "发动机动力",
        "方向盘手感",
        "座椅舒适度",
        "车身设计"
    ]
    
    for need in test_needs:
        print(f"\n需求: {need}")
        result = classify_need(need, prompt_template)
        if result:
            parsed_result = result.get("parsed_result")
            raw_response = result.get("raw_response", "")
            print(f"分类结果: {parsed_result}")
            print(f"LLM原始响应: {raw_response}")
        else:
            print("分类结果: None")
        print("-" * 50)

if __name__ == "__main__":
    global start_time  # 声明要使用全局变量
    print("喵呜~异步需求分类系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 记录启动时间
    start_time = time.time()
    
    # 演示单条分析
    # demo_single_analysis()
    
    # 异步批量分析评论
    asyncio.run(batch_analyze_comments_async(
        csv_file_path='results/sentiment_analysis_results.csv',  # 使用第一步的输出作为输入
        output_file_path='results/classification_results.csv',
        batch_size=1,  # 每批1条
        max_concurrent=90,  # 最大并发数
        cooldown=1  # 每条处理后等待
    ))
    
    print("\n任务完成喵~尾巴高速摇摆中！ヽ(=^･ω･^=)丿")