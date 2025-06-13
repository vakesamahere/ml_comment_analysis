#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车评论情感分析测试脚本 🚗💖
用于验证系统功能是否正常工作喵~
"""

from tuple_notation import load_sentiment_prompt, analyze_comment_sentiment, parse_sentiment_result

def test_prompt_loading():
    """测试提示词加载功能"""
    print("=== 测试提示词加载 ===")
    prompt = load_sentiment_prompt()
    if prompt:
        print("✅ 提示词加载成功喵~")
        print(f"提示词长度: {len(prompt)} 字符")
        return True
    else:
        print("❌ 提示词加载失败喵~")
        return False

def test_result_parsing():
    """测试结果解析功能"""
    print("\n=== 测试结果解析 ===")
    
    test_cases = [
        "('加速性能', '积极')",
        '("导航系统", "消极")',
        "回复: ('车内空间', '中性')",
        "分析结果：('音响效果', '积极')",
        "无效的返回格式"
    ]
    
    for case in test_cases:
        result = parse_sentiment_result(case)
        print(f"输入: {case}")
        print(f"解析结果: {result}")
        print("-" * 30)

def test_sample_comments():
    """测试示例评论分析（不需要真实API）"""
    print("\n=== 测试示例评论分析 ===")
    
    # 模拟一些API返回结果来测试解析功能
    mock_responses = [
        "('加速性能', '积极')",
        "('导航系统', '消极')", 
        "('车内空间', '中性')"
    ]
    
    sample_comments = [
        "这辆车的加速性能真是太棒了",
        "导航系统经常卡顿，很困扰",
        "车内空间还可以，不算大但也不挤"
    ]
    
    for i, (comment, mock_response) in enumerate(zip(sample_comments, mock_responses)):
        print(f"评论 {i+1}: {comment}")
        result = parse_sentiment_result(mock_response)
        print(f"解析结果: {result}")
        print("-" * 40)

def main():
    """主测试函数"""
    print("喵呜~汽车评论情感分析系统测试开始啦！ฅ^•ﻌ•^ฅ\n")
    
    # 测试各个功能模块
    prompt_ok = test_prompt_loading()
    test_result_parsing()
    test_sample_comments()
    
    print("\n=== 测试总结 ===")
    if prompt_ok:
        print("✅ 基础功能测试通过喵~")
        print("💡 提示: 配置好.env文件后可以进行完整的API测试")
    else:
        print("❌ 发现问题，请检查prompts目录和文件")
    
    print("\n尾巴摇摆中~测试完成啦！ヽ(=^･ω･^=)丿")

if __name__ == "__main__":
    main()