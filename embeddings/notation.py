import pandas as pd
import csv
import os
import asyncio
import aiohttp
import json
import time
from tqdm.asyncio import tqdm as async_tqdm
from datetime import timedelta

# 全局变量
failure_count = 0   # 失败次数
total_processed = 0 # 已处理总数
total_time = 0     # 总处理时间

class OllamaEmbeddingClient:
    """
    Ollama Embedding 客户端喵~
    """
    def __init__(self, base_url="http://localhost:11434", model_name="dengcao/Qwen3-Embedding-0.6B:Q8_0"):
        self.base_url = base_url
        self.model_name = model_name
        self.embedding_url = f"{base_url}/api/embeddings"
    
    async def get_embedding(self, text):
        """
        获取文本的embedding向量喵~
        
        参数:
        - text: 要处理的文本
        
        返回:
        - list: embedding向量列表，如果失败返回None
        """
        if not text or not text.strip():
            return None
        
        payload = {
            "model": self.model_name,
            "prompt": text.strip()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.embedding_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result.get('embedding')
                        if embedding and isinstance(embedding, list):
                            return embedding
                        else:
                            print(f"咪啾~embedding字段格式不正确: {result}")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"咪啾~API请求失败 {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            print(f"咪啾~获取embedding时出错: {str(e)} (´･ω･`)")
            return None

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
            if result['embedding'] is None:
                deletions.append(result)
                failure_count += 1
                if pbar:
                    pbar.set_postfix({'失败': failure_count}, refresh=True)
                continue
            
            # 将embedding列表转换为字符串存储
            if isinstance(result['embedding'], list):
                result['embedding'] = json.dumps(result['embedding'])
        
        results_batch = [result for result in results_batch if result not in deletions]
        
        # 使用标准csv模块写入
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['global_id', 'topic_id', 'reply_id', 'user_id', 'region', 'time', 'content', 'embedding']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for result in results_batch:
                writer.writerow(result)
                
    except Exception as e:
        print(f"咪啾~保存CSV文件失败: {str(e)} (´･ω･`)")

async def process_comment_batch(batch_data, embedding_client, output_file_path, pbar=None):
    """
    处理一批评论数据，生成embedding喵~（异步版本）
    """
    results_batch = []
    
    for _, row in batch_data.iterrows():
        global_id = str(row.get('global_id', ''))
        topic_id = str(row.get('topic_id', ''))
        reply_id = str(row.get('reply_id', ''))
        user_id = str(row.get('user_id', ''))
        region = str(row.get('region', ''))
        time_str = str(row.get('time', ''))
        content = str(row.get('content', '')).strip()
        
        if not content or content == 'nan':
            continue
        
        # 获取embedding
        embedding = await embedding_client.get_embedding(content)
        
        # 保存结果
        result = {
            'global_id': global_id,
            'topic_id': topic_id,
            'reply_id': reply_id,
            'user_id': user_id,
            'region': region,
            'time': time_str,
            'content': content,
            'embedding': embedding
        }
        results_batch.append(result)
    
    # 批量保存到CSV（传递进度条）
    await save_batch_results_to_csv(results_batch, output_file_path, pbar)
    
    return len(results_batch)

async def batch_generate_embeddings_async(csv_file_path, output_file_path=None, length=-1, batch_size=1, max_concurrent=1, cooldown=1):
    """
    异步批量生成评论embedding喵~
    
    参数:
    - csv_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - length: 处理的评论数量限制
    - batch_size: 每批处理的评论数量
    - max_concurrent: 最大并发数
    - cooldown: 处理间隔（秒）
    """
    print("喵呜~异步embedding生成系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 创建embedding客户端
    embedding_client = OllamaEmbeddingClient()
    
    # 设置输出文件路径
    if not output_file_path:
        output_file_path = 'results/embeddings/embed_notation.csv'
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 读取CSV数据
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 条数据喵~ (≧∇≦)ﾉ")
        
        # 检查已处理的数据
        processed_ids = []
        if os.path.exists(output_file_path):
            try:
                processed_df = pd.read_csv(output_file_path, encoding='utf-8')
                processed_ids = processed_df['reply_id'].astype(str).tolist()
            except:
                processed_ids = []
        
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
    pbar = async_tqdm(total=len(df), desc="生成embedding中", unit="条")
    
    async def process_with_semaphore(batch):
        global total_processed, total_time
        
        async with semaphore:
            batch_start_time = asyncio.get_event_loop().time()
            result = await process_comment_batch(batch, embedding_client, output_file_path, pbar)
            
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
    
    print(f"批量embedding生成完成！共处理 {processed_count} 条数据喵~ (ΦωΦ)")
    print(f"总用时: {total_time_mins:.1f}分钟")
    print(f"平均速度: {avg_speed:.1f}条/分钟")
    print(f"失败数: {failure_count} 条 (失败率: {failure_rate:.1f}%) (｡•́︿•̀｡)")
    print(f"结果已保存到 {output_file_path} 喵~ ✧(≖ ◡ ≖✿)")
    
    return processed_count

async def demo_single_embedding():
    """
    演示单条embedding生成喵~
    """
    print("=== 单条embedding生成演示 ===")
    
    embedding_client = OllamaEmbeddingClient()
    
    # 示例文本
    test_texts = [
        "啥价",
        "健康度真不错，现在用LG的三元锂肯定不如之前的松下",
        "Y总感觉颠，所以我也选了3，车一年也带不了几次人，越买越小了",
        "看了下第一年保险，还是有点小贵"
    ]
    
    for text in test_texts:
        print(f"\n文本: {text}")
        embedding = await embedding_client.get_embedding(text)
        if embedding:
            print(f"Embedding维度: {len(embedding)}")
            print(f"前5个值: {embedding[:5]}")
        else:
            print("Embedding生成失败")
        print("-" * 50)

if __name__ == "__main__":
    global start_time  # 声明要使用全局变量
    print("喵呜~异步embedding生成系统启动啦！ฅ^•ﻌ•^ฅ")
    
    # 记录启动时间
    start_time = time.time()
    
    # 演示单条embedding生成
    # asyncio.run(demo_single_embedding())
    
    # 异步批量生成embedding
    asyncio.run(batch_generate_embeddings_async(
        csv_file_path='data/comment_contents_cleaned.csv',  # 使用指定的输入文件
        output_file_path='results/embeddings/embed_notation.csv',  # 使用指定的输出文件
        batch_size=1,  # 每批1条
        max_concurrent=150,  # 并发数设为1
        cooldown=0  # 每条处理后等待1秒
    ))
    
    print("\n任务完成喵~尾巴高速摇摆中！ヽ(=^･ω･^=)丿")