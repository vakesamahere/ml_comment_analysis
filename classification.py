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

def load_classification_prompt(prompt_file='prompts/classification.txt'):
    """
    加载分类提示词模板喵~
    
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
            if result.get('classification') is None:
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
        output_columns = [id_column, content_column, 'requirement', 'sentiment', 'classification', 'llm_raw_response']
    
    for _, row in batch_data.iterrows():
        # 获取ID，如果不存在则使用索引作为ID
        item_id = str(row.get(id_column, '')) if id_column in row else str(_)
        content = str(row.get(content_column, '')).strip()
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
            id_column: item_id,
            content_column: content,
            'requirement': requirement,
            'sentiment': sentiment,
            'classification': parsed_result,
            'llm_raw_response': raw_response
        }
        
        # 添加原始数据中的其他列
        for col in row.index:
            if col not in [id_column, content_column, 'requirement', 'sentiment'] and col not in result:
                result[col] = row[col]
                
        results_batch.append(result)
    
    # 批量保存到CSV（传递进度条）
    await save_batch_results_to_csv(results_batch, output_file_path, output_columns, pbar)
    
    return len(results_batch)

async def batch_analyze_comments_async(csv_file_path, output_file_path=None, length=-1, batch_size=1, 
                                     max_concurrent=25, cooldown=2, id_column='reply_id', 
                                     content_column='content', extra_columns=None, 
                                     prompt_file=None, auto_include_columns=True,**config):
    """
    异步批量分析评论数据喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - length: 处理的评论数量限制
    - batch_size: 每批处理的评论数量
    - max_concurrent: 最大并发数
    - cooldown: 每条处理后的冷却时间（秒）
    - id_column: ID列的列名
    - content_column: 内容列的列名
    - extra_columns: 需要保留的额外列名列表
    - prompt_file: 自定义提示词文件路径
    - auto_include_columns: 是否自动包含所有原始数据列
    """
    print("喵呜~异步评论分类系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 加载提示词模板
    prompt_template = load_classification_prompt(prompt_file) if prompt_file else load_classification_prompt()
    if not prompt_template:
        return []
    
    # 设置输出文件路径
    if not output_file_path:
        output_file_path = 'results/classification_results.csv'
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 确定要保留的列
    if extra_columns is None:
        extra_columns = []
        
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条数据喵~ (≧∇≦)ﾉ")
        
        # 确保必要的列存在
        required_cols = [id_column, content_column, 'requirement', 'sentiment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"咪啾~找不到必要的列: {', '.join(missing_cols)}! (´･ω･`)")
            if 'requirement' in missing_cols or 'sentiment' in missing_cols:
                print("缺少必要的需求和情感列，无法进行分类分析(｡•́︿•̀｡)")
                return []
            
        # 如果ID列不存在，创建一个索引作为ID列
        if id_column not in df.columns:
            print(f"找不到ID列 '{id_column}'，将使用索引作为ID喵~")
            df[id_column] = df.index.astype(str)
        
        # 自动包含所有列作为输出列
        if auto_include_columns:
            for col in df.columns:
                if col not in [id_column, content_column, 'requirement', 'sentiment'] and col not in extra_columns:
                    extra_columns.append(col)
            print(f"自动包含了 {len(extra_columns)} 个额外列喵~ ✧(≖ ◡ ≖✿)")
    
        output_columns = [id_column, content_column, 'requirement', 'sentiment', 'classification', 'llm_raw_response'] + extra_columns
        
        # 获取已处理的ID
        processed_ids = set()
        if os.path.exists(output_file_path):
            try:
                processed_df = pd.read_csv(output_file_path, encoding='utf-8')
                if id_column in processed_df.columns:
                    processed_ids = set(processed_df[id_column].astype(str))
                print(f"喵呜~发现已处理的评论 {len(processed_ids)} 条！继续上次的工作 ฅ^•ﻌ•^ฅ")
            except Exception as e:
                print(f"咪啾~读取历史结果文件出错: {str(e)} (´･ω･`)")
        
        # 过滤已处理的数据
        if processed_ids:
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

def clear_illegal_types(**config):
    """
    清理DataFrame中指定列的非法类型喵~
    
    参数:
    - df: 要处理的DataFrame
    - columns: 要清理的列名列表
    """
    df = pd.read_csv(config.get('output_file_path', ''), encoding='utf-8')
    classification_types = config.get('classification_types', [])
    output_path = config.get('output_file_path', 'results/classification_results.csv')
    for index, row in df.iterrows():
        classification = row.get('classification', '')
        if isinstance(classification, str):
            # 使用正则表达式匹配合法类型
            if not classification in classification_types:
                print(f"清理非法类型: {classification} 在索引 {index} 处被标记为无关喵~")
                df.at[index, 'classification'] = '无关'
                # input()
    df.to_csv(output_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    global start_time  # 声明要使用全局变量
    print("喵呜~异步需求分类系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 记录启动时间
    start_time = time.time()
    
    # 自定义配置
    config = {
        'csv_file_path': 'case_study/mtr/results/MTR_demand_tuple.csv',  # 使用第一步的输出作为输入
        'output_file_path': 'case_study/mtr/results/MTR_demand_tuple_classified.csv',   # 分类结果输出路径
        'length': -1,          # -1表示处理所有评论
        'batch_size': 1,       # 每批x条
        'max_concurrent': 90,  # 最大并发数
        'cooldown': 1,         # 每条评论处理后等待z秒
        'id_column': 'id',      # ID列的列名
        'content_column': 'Content',  # 内容列的列名
        'extra_columns': [],   # 手动指定需要保留的额外列
        'prompt_file': 'case_study/mtr/prompts/classification.txt',   # 自定义提示词文件，None表示使用默认
        'auto_include_columns': False,  # 自动包含所有原始列
        'classification_types': ["等待时间","APP","产品体验","营销活动","环境氛围","服务体验","卫生情况","地理位置"]
    }
    
    # 演示单条分析
    # demo_single_analysis()
    
    # 异步批量分析评论
    asyncio.run(batch_analyze_comments_async(**config))
    
    print("\n任务完成喵~尾巴高速摇摆中！ヽ(=^･ω･^=)丿")

    if 1:
        # 清理非法类型
        clear_illegal_types(**config)
        print("清理非法类型完成喵~ (≧∇≦)ﾉ")