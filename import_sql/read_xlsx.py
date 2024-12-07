import pandas as pd


def load_excel_data(file_path):
    data = pd.read_excel(file_path)
    data['IdealQuestionId'] = data['IdealQuestionId']
    data['IsDelete'] = data['IsDelete'].apply(lambda x: True if x == 1 else False)  # 将"IsDelete"列处理为空值为False，1为True

    return data
