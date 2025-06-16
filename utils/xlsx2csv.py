# 读取xlsx，转换成csv
import pandas as pd
import numpy as np

xlsx_path = 'case_study/mtr/data/MTR.xlsx'
csv_path = 'case_study/mtr/data/MTR.csv'

def xlsx_to_csv(xlsx_path, csv_path):
    """
    将xlsx文件转换为csv文件，并将所有单元格中的换行符替换为空格
    :param xlsx_path: 输入的xlsx文件路径
    :param csv_path: 输出的csv文件路径
    """
    try:
        # 读取xlsx文件
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        
        # 替换所有列中的换行符为空格
        for column in df.columns:
            # 只处理字符串类型的列
            if df[column].dtype == object:
                df[column] = df[column].replace(to_replace=r'\n|\r\n|\r', value=' ', regex=True)
        
        # 将DataFrame保存为csv文件
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"成功将 {xlsx_path} 转换为 {csv_path}，并将换行符替换为空格")
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    xlsx_to_csv(xlsx_path, csv_path)