import unittest
import os
import csv
import asyncio
from tuple_notation import save_batch_results_to_csv

class TestCSVWriter(unittest.TestCase):
    def setUp(self):
        self.test_file = 'test_output.csv'
        # 确保每次测试前文件不存在
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        # 测试后清理文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_csv_write_with_commas(self):
        # 准备测试数据，包含逗号的内容
        test_batch = [
            {
                'reply_id': '1',
                'content': 'This has, a comma',
                'requirement': 'Needs, more, features',
                'sentiment': 'positive, very good',
                'llm_raw_response': 'Response with, many, commas'
            },
            {
                'reply_id': '2',
                'content': 'Text with "quotes", and, commas',
                'requirement': 'Another, requirement',
                'sentiment': 'negative',
                'llm_raw_response': 'Raw, response'
            }
        ]

        # 执行异步写入
        asyncio.run(save_batch_results_to_csv(test_batch, self.test_file))

        # 验证写入的内容
        with open(self.test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # 验证行数
            self.assertEqual(len(rows), 2)

            # 验证第一行数据
            self.assertEqual(rows[0]['content'], 'This has, a comma')
            self.assertEqual(rows[0]['requirement'], 'Needs, more, features')

            # 验证第二行数据
            self.assertEqual(rows[1]['content'], 'Text with "quotes", and, commas')

if __name__ == '__main__':
    unittest.main()