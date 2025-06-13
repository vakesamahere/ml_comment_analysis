import pandas
import json
df_comment_data = pandas.read_csv('data/comment_contents_cleaned.csv')
df_classification = pandas.read_csv('results/classification_results.csv')
# 对齐type
legal_classifications = json.load(open('prompts/types.json', 'r', encoding='utf-8'))
df_classification['classification'] = df_classification['classification'].apply(
    lambda x: x if x in legal_classifications else '其他'
)

# 对齐sentiment
map = {
    "积极":"积极",
    "消极":"消极",
    "中性":"中性",
    "无":"中性",
    "不适用":"中性",
    "情感":"中性",
    "未识别情感":"中性",
    "无法提取":"中性",
    "无法分析":"中性",
    "无法确定":"中性",
    "emotion":"中性",
    "无法识别":"中性",
    "未知":"中性",
    "后悔":"消极",
    "纠结":"消极",
    "无法识别的情感":"中性",
    "无情感":"中性",
    "不确定":"中性",
    "缺失":"中性",
    "无法判断":"中性",
    "疑问":"中性",
    "積極":"积极",
    "消极?":"中性",
    "无相关情感":"中性",
    "建议":"中性",
    "期待":"积极",
    "需求":"中性",
    "积极/消极/中性":"中性",
}
df_classification['sentiment'] = df_classification['sentiment'].apply(lambda x: map.get(x, '中性'))

df_user_data = df_comment_data[['reply_id', 'user_id', 'region', 'time']].drop_duplicates()
# connect when df_comment_data['reply_id'] == df_user_data['reply_id']
# 以df_classification行数为准
df_user_data = df_user_data.merge(df_classification[['reply_id', 'classification', 'sentiment']], on='reply_id', how='right')
# to csv
df_user_data.to_csv('results/classification_with_user.csv', index=False)