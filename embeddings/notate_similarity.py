import pandas as pd
import csv
import os
import asyncio
import aiohttp
import json
import time
from tqdm.asyncio import tqdm as async_tqdm
from datetime import timedelta

class OllamaEmbeddingClient:
    """
    Ollama Embedding 客户端喵~
    """
    def __init__(self,
                 base_url="http://localhost:11434",
                 model_name="dengcao/Qwen3-Embedding-0.6B:Q8_0",
                 endpoint="api/embeddings",
                 input_key = "prompt",
                 output_struct: list[str] = ["embedding"],
                 api_key=""
                 ):
        self.base_url = base_url
        self.model_name = model_name
        self.embedding_url = f"{base_url}/{endpoint}"
        self.output_struct = output_struct
        self.input_key = input_key
        self.api_key = api_key
    
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
            self.input_key: text.strip()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.embedding_url,
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
                        }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result
                        for key in self.output_struct:
                            k = key.split('/')
                            if len(k) == 2: # like "data/0"
                                key_ = k[0]
                                index = int(k[1])
                                embedding = embedding.get(key_, [None])
                                if isinstance(embedding, list) and len(embedding) > index:
                                    embedding = embedding[index]
                                else:
                                    # print(f"咪啾~未找到输出结构中的字段: {key} 或索引: {index}")
                                    return None
                            else:  # like "embedding"
                                embedding = embedding.get(key, None)
                            if embedding is None:
                                # print(f"咪啾~未找到输出结构中的字段: {key}")
                                return None
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
        
client = OllamaEmbeddingClient(
    base_url="https://oneapi.ainet.eu.org/v1",
    model_name="text-embedding-3-large",
    endpoint="embeddings",
    input_key="input",
    output_struct=["data/0","embedding"],
    api_key="sk-B6jgoJPQjZ1H5fr_OmqLZNLSVZFn7fiShGQVAAvChCpQAMxtNOgaE_-NOXI"  # 如果需要API密钥，请在这里填写喵~
)

def get_embedding_sync(text: str, client: OllamaEmbeddingClient = client):
    """同步获取嵌入向量的包装函数"""
    return asyncio.run(client.get_embedding(text))

def get_cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度喵~
    
    参数:
    - vec1: 第一个向量
    - vec2: 第二个向量
    
    返回:
    - float: 余弦相似度
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量长度不一致喵~")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a ** 2 for a in vec1) ** 0.5
    norm_b = sum(b ** 2 for b in vec2) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

demensions = [
  "驾驶感受",
  "操控",
  "舒适性",
  "外观",
  "内饰",
  "性价比",
  "续航能力",
  "智能"
]

pack_funs = [
  lambda x:f"用户已经表明，或有认为{x}很好的倾向",
  lambda x:f"用户提到了{x}，但是没有表示好或者不好",
  lambda x:f"用户已经表明，或有认为{x}不好的倾向",
  lambda x:f"用户在对某车辆的评论中，说到了{x}的相关信息",
  lambda x:f"用户在对某车辆的评论中，没有说到{x}的相关信息",
]

pack_user_comment = lambda x: f"{x}"

incredible_check_funs = [pack_funs[0], pack_funs[1], pack_funs[2]]
mention_check_funs = [pack_funs[-2], pack_funs[-1]]
polarity_embds = lambda x, funs: [map[f(x)] for f in funs]
polarity_sims = lambda x, tup: [get_cosine_similarity(x, tup[0]), get_cosine_similarity(x, tup[1])]
check_positive = lambda x, tup: get_cosine_similarity(x, tup[0]) > get_cosine_similarity(x, tup[1])
check_max_index = lambda x, tup: max(enumerate([get_cosine_similarity(x, tup[i]) for i in range(len(tup))]), key=lambda item: item[1])[0]

# try to read existing embeddings from file
if os.path.exists("results/embeddings/embed_map.json"):
    with open("results/embeddings/embed_map.json", "r", encoding='utf-8') as f:
        map = json.load(f)
# 如果没有文件，则创建一个空的字典并创建文件
else:
    map = {}
    with open("results/embeddings/embed_map.json", "w", encoding='utf-8') as f:
        json.dump(map, f, ensure_ascii=False, indent=4)
for demension in demensions:
    for sentiment in pack_funs:
        if sentiment(demension) in map:
            continue
        embed = get_embedding_sync(sentiment(demension))
        map[sentiment(demension)] = embed
        if embed is None or embed == []:
            time.sleep(10)
            continue
        with open("results/embeddings/embed_map.json", "w", encoding='utf-8') as f:
            json.dump(map, f, ensure_ascii=False, indent=4)
            print(f"已更新嵌入向量: {sentiment(demension)}")
        time.sleep(1)


# ====

def get_tuple_notation(comment)-> tuple[str, str]:
    global map, client, pack_user_comment
    res = {}
    
    comment = pack_user_comment(comment)

    comment_embed = get_embedding_sync(comment)
    demension_mentioned = []
    for demension in demensions:
        embd_mentioned = polarity_embds(demension, mention_check_funs)
        if check_positive(comment_embed, embd_mentioned):
            demension_mentioned.append([demension, embd_mentioned[0]])

    if not demension_mentioned:
        # print(f"没有提及任何维度喵~")
        return ('无','中性')
    # x[1]最大的x[0]是most_mentioned
    most_mentioned = max(demension_mentioned, key=lambda x: x[1], default=(None, None))[0]
    check_positive_flag = check_max_index(comment_embed, polarity_embds(most_mentioned, incredible_check_funs))
    incredible_map = {
        0: "积极",
        1: "中性",
        2: "消极",
    }
    return (most_mentioned, incredible_map[check_positive_flag])

def is_mentioned(demension, comment_embd):
    """
    检查评论中是否提到了某个维度喵~
    
    参数:
    - demension: 维度名称
    - comment_embd: 评论的embedding向量
    
    返回:
    - bool: 是否提到该维度
    """
    tup = polarity_embds(demension, mention_check_funs)
    return check_positive(comment_embd, tup)

def is_good_comment(demension, comment_embd):
    """
    检查评论是否对某个维度有正面评价喵~
    
    参数:
    - demension: 维度名称
    - comment_embd: 评论的embedding向量
    
    返回:
    - bool: 是否对该维度有正面评价
    """
    tup = polarity_embds(demension, incredible_check_funs)
    return check_positive(comment_embd, tup)

def test():
    examples = [
      "确实晕车，以前坐我朋友的特斯拉回回都晕车",
      "好像是有晕车的情况，不过一般开的人不晕，是坐的人晕，电车有这现象。",
      "车开得很舒服啊",
      "大神们都来说一说，反馈一下提车时间"
    ]
    for comment in examples:
        print(f"处理评论: {comment}")
        evidence_list = get_tuple_notation(comment)
        print(f"{evidence_list}")

def main(cd=0):
    df = pd.read_csv("results/embeddings/embed_notation.csv", encoding='utf-8').head(10)
    df_res = pd.DataFrame(columns=['reply_id', 'comment', 'requirement', 'sentiment'])
    for index, row in df.iterrows():
        content = row['content']
        reply_id = row['reply_id']
        # comment_embd:str = row['embedding'] # "[0.1,0.2]"
        # comment_embd = json.loads(comment_embd)
        print(f"处理评论: {content} (回复ID: {reply_id})")

        tup = get_tuple_notation(content)
        df_temp = pd.DataFrame({
            'reply_id': [reply_id],
            'comment': [content],
            'requirement': [tup[0]],
            'sentiment': [tup[1]]
        })
        df_res = pd.concat([df_res, df_temp], ignore_index=True)
        df_res.to_csv("results/embeddings/tuple_notation.csv", index=False, encoding='utf-8')
        if cd > 0:
            time.sleep(cd)
    df_res.to_csv("results/embeddings/tuple_notation.csv", index=False, encoding='utf-8')


main(cd=1)