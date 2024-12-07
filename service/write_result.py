import pandas as pd
from openpyxl import Workbook
import os
from utils import rating_path


def write_result(data):
    # 如果文件存在，则清空数据
    if os.path.exists(rating_path):
        # 创建一个空的工作簿
        wb = Workbook()
        wb.save(rating_path)

    # 将 DataFrame 写入 Excel 文件
    with pd.ExcelWriter(rating_path, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="Sheet1")

    print(f"数据已成功写入 {rating_path}")
