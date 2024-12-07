import json
import pandas as pd


def load_json_data(file_path):
    # with open(file_path, 'rb') as f:
    #     raw_data = f.read()
    #     result = chardet.detect(raw_data)
    #     encoding = result['encoding']

    # with open(file_path, encoding=encoding, errors='ignore') as f:
    #     json_data = json.load(f)

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data = []

    for item in json_data:
        contextId = item['context_id']  # 驼峰命名法
        context = item['context_text']
        qas = item['qas']

        # 遍历每个 "qas" 下的 query_text 和 answers
        for qa in qas:
            query_text = qa['query_text']
            answers = qa['answers']
            data.append([contextId, context, query_text, answers])

    data = pd.DataFrame(data, columns=['ContextId', 'Context', 'Question', 'Answers'])
    print("数据读取完成")
    return data
