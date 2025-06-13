import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 中文显示设置 - 修复字体问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'  # 黑体
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# # 设置图表样式
# try:
#     plt.style.use('seaborn-v0_8')
# except:
#     plt.style.use('seaborn')
# sns.set_palette("husl")

# 读取数据
print("🐱 喵~ 正在读取数据...")
df = pd.read_csv('results/classification_with_user.csv')

# 确保输出目录存在
output_dir = 'results/classification_stats/'
os.makedirs(output_dir, exist_ok=True)

print(f"🐱 数据概览：共有 {len(df)} 条记录喵~")
print(f"🐱 涉及 {df['region'].nunique()} 个省份")
print(f"🐱 涉及 {df['classification'].nunique()} 个分类")
print(f"🐱 情感分布：{df['sentiment'].value_counts().to_dict()}")

# ==================== 任务1：省份比例和需求权重分析 ====================

print("\n🐱 任务1：分析省份比例和需求权重...")

# 1.1 省份评论数量统计
region_stats = df.groupby('region').size().reset_index(name='total_comments')
region_stats = region_stats.sort_values('total_comments', ascending=False)
region_stats['percentage'] = (region_stats['total_comments'] / region_stats['total_comments'].sum() * 100).round(2)

print("🐱 省份评论数量排行榜（前10名显示）：")
for _, row in region_stats.head(10).iterrows():
    print(f"   {row['region']}: {row['total_comments']}条 ({row['percentage']}%)")
print(f"🐱 完整数据已保存到CSV文件（共{len(region_stats)}个省份）")

# 1.2 各省份对不同需求的关注度
classification_by_region = df.groupby(['region', 'classification']).size().unstack(fill_value=0)
classification_percentage = classification_by_region.div(classification_by_region.sum(axis=1), axis=0) * 100

# 保存详细统计数据
region_stats.to_csv(f'{output_dir}region_statistics.csv', index=False, encoding='utf-8-sig')
classification_percentage.to_csv(f'{output_dir}region_classification_percentage.csv', encoding='utf-8-sig')

# 生成省份评论数量柱状图
print("🐱 生成省份评论数量图表...")
plt.figure(figsize=(14, 8))

# 图表只显示前15个省份
region_stats_for_chart = region_stats.head(15)
bars = plt.bar(range(len(region_stats_for_chart)),
               region_stats_for_chart['total_comments'],
               color=plt.cm.viridis(np.linspace(0, 1, len(region_stats_for_chart))))

plt.title('各省份评论数量统计\n(前15个省份)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('省份', fontsize=12, fontweight='bold')
plt.ylabel('评论数量', fontsize=12, fontweight='bold')
plt.xticks(range(len(region_stats_for_chart)),
           region_stats_for_chart['region'],
           rotation=45, ha='right')

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}region_statistics_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("🐱 各省份需求关注度分析已保存~")
print("🐱 省份评论数量图表已生成~")

# ==================== 任务2&3：堆叠柱状图（4种情感类型）====================

print("\n🐱 任务2&3：生成堆叠柱状图...")

def create_stacked_bar_chart(data_subset, title, filename, sentiment_filter=None):
    """创建堆叠柱状图"""
    
    if sentiment_filter:
        filtered_data = data_subset[data_subset['sentiment'] == sentiment_filter]
        if len(filtered_data) == 0:
            print(f"   ⚠️ {sentiment_filter}情感数据为空，跳过图表生成")
            return
    else:
        filtered_data = data_subset
    
    # 统计各省各分类的数量
    pivot_data = filtered_data.groupby(['region', 'classification']).size().unstack(fill_value=0)
    
    # 按评论数量排序
    region_totals = pivot_data.sum(axis=1).sort_values(ascending=False)
    pivot_data_sorted = pivot_data.loc[region_totals.index]
    
    # 图表只显示前15个省份
    pivot_data_for_chart = pivot_data_sorted.head(15)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制堆叠柱状图
    pivot_data_for_chart.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    
    plt.title(f'{title}\n(前15个省份)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('省份', fontsize=12, fontweight='bold')
    plt.ylabel('评论数量', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='分类类型', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存对应的数据（全部省份数据）
    pivot_data_sorted.to_csv(f'{output_dir}{filename.replace(".png", "_data.csv")}', encoding='utf-8-sig')
    
    print(f"   ✅ {title} 图表已生成：{filename}")

# 生成4种类型的图表
chart_configs = [
    (df, "各省份各需求评论总量分布", "total_comments_by_region.png", None),
    (df, "各省份各需求积极评论分布", "positive_comments_by_region.png", "积极"),
    (df, "各省份各需求中立评论分布", "neutral_comments_by_region.png", "中性"),
    (df, "各省份各需求消极评论分布", "negative_comments_by_region.png", "消极")
]

for data, title, filename, sentiment in chart_configs:
    create_stacked_bar_chart(data, title, filename, sentiment)

# ==================== 额外分析：需求关注度饼图 ====================

print("\n🐱 额外福利：生成需求关注度分析...")

# 整体需求分布饼图
classification_counts = df['classification'].value_counts()

plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(classification_counts)))
wedges, texts, autotexts = plt.pie(classification_counts.values,
                                  labels=classification_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90)

plt.title('整体需求关注度分布', fontsize=16, fontweight='bold', pad=20)

# 美化文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{output_dir}overall_classification_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n🐱 喵呜~ 所有任务完成啦！ ฅ^•ﻌ•^ฅ")
print(f"🐱 所有图表和数据已保存到：{output_dir}")
print("🐱 生成的文件包括：")
print("   📊 各省份评论分布图表（总量、积极、中立、消极）")
print("   📈 整体需求关注度饼图")
print("   📋 情感分析概览图")
print("   📄 详细统计数据CSV文件")
print("\n🐱 任务完成，主人可以查看结果了喵~ (◕‿◕)✨")
