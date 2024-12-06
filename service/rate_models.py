import time
import numpy as np
import pandas as pd
from numpy.ma.extras import average
import pickle
from scipy.spatial.distance import cosine
from utils.db_connect import connect
from import_sql.import_ques import import_ques
from service import weights_path
from service.read_sql import read_trial, read_ques
from service.write_result import write_result

NUM_SIMILARITY = 11  # trial中最接近的问题和模型判断的(K-1)个近似问题进行比较
ALPHA, BETA = 1, 0.25
P = -1

# 各模型评价
def rate(ques_info, trial_info):
    def rate_vec(vec, ques, ideal_ques_id, list_embedding, model_name):
        item = 0
        ideal_ques = ''
        similarity_list = []  # 最相关的K个向量的相似度
        temp_ques_id = 1

        for embedding in list_embedding:
            similarity = 1 - cosine(vec, embedding)  # 计算余弦相似度
            if len(similarity_list) == 0:
                similarity_list.append(similarity)
            elif len(similarity_list) < NUM_SIMILARITY:
                if similarity > max(similarity_list):
                    temp_ques_id = trial_info.loc[item, 'Id']  # 记录similarity最大值对应Id
                similarity_list.append(similarity)
            else:
                if similarity > min(similarity_list):
                    if similarity > max(similarity_list):
                        temp_ques_id = trial_info.loc[item, 'Id']  # 记录similarity最大值对应Id
                    similarity_list[similarity_list.index(min(similarity_list))] = similarity

            item += 1

        if ideal_ques_id != temp_ques_id:
            guess_ques = trial_info.loc[trial_info['Id'] == temp_ques_id, 'Question'].values[0]
            ideal_ques = trial_info.loc[trial_info['Id'] == ideal_ques_id, 'Question'].values[0]
            print("模型"+model_name+" 用户问题："+ques+" 误判问题："+guess_ques+" 真实问题："+ideal_ques)
            return P

        similarity_list.sort(reverse=True)  # 按照相似度从大到小排序
        result = round(ALPHA * similarity_list[0] - BETA * average(similarity_list[1: ]), 4)
        return result

    def cal_stella(row):
        vec = row['StellaEmbedding']
        ques = row['Question']
        ideal_ques_id = row['IdealQuestionId']

        list_stella = list(trial_info['StellaEmbedding'])
        return rate_vec(vec, ques, ideal_ques_id, list_stella, 'Stella')

    def cal_gte(row):
        vec = row['GteEmbedding']
        ques = row['Question']
        ideal_ques_id = row['IdealQuestionId']

        list_gte = list(trial_info['GteEmbedding'])
        return rate_vec(vec, ques, ideal_ques_id, list_gte, 'Gte')

    def cal_dmeta(row):
        vec = row['DmetaEmbedding']
        ques = row['Question']
        ideal_ques_id = row['IdealQuestionId']

        list_dmeta = list(trial_info['DmetaEmbedding'])
        return rate_vec(vec, ques, ideal_ques_id, list_dmeta, 'Dmeta')

    ques_info['StellaRating'] = ques_info.apply(cal_stella, axis=1)
    ques_info['GteRating'] = ques_info.apply(cal_gte, axis=1)
    ques_info['DmetaRating'] = ques_info.apply(cal_dmeta, axis=1)

    # 删除原始数据中的Embedding
    ques_info = ques_info.drop(columns=['StellaEmbedding', 'GteEmbedding','DmetaEmbedding'])
    excel_info = ques_info.copy()

    # 计算每列的平均值
    stella_avg = round(excel_info['StellaRating'].mean(), 4)
    gte_avg = round(excel_info['GteRating'].mean(), 4)
    dmeta_avg = round(excel_info['DmetaRating'].mean(), 4)

    excel_info.loc[len(excel_info)] = [np.nan] * len(excel_info.columns)
    excel_info.at[len(excel_info) - 1, 'Question'] = 'Average'  # 修改最后一行的 'Question' 列为 "Average"

    # 在每列的最后一行添加相应的平均值
    excel_info.at[len(excel_info) - 1, 'StellaRating'] = stella_avg
    excel_info.at[len(excel_info) - 1, 'GteRating'] = gte_avg
    excel_info.at[len(excel_info) - 1, 'DmetaRating'] = dmeta_avg

    write_result(excel_info)

    return ques_info

# 函数用于归一化处理
def normalize_ratings(row):
    ratings = row[['StellaRating', 'GteRating', 'DmetaRating']].values
    
    # 如果三列都有非-1值
    if all(r != -1 for r in ratings):
        # 找到最小值和最大值
        min_val = min(ratings)
        max_val = max(ratings)
        
        # 归一化中间值
        if max_val != min_val:
            normalized = [(r - min_val) / (max_val - min_val) if r != max_val else 1 for r in ratings]
        else:
            normalized = ratings  # 如果最大最小相等，不做变化
            
    else:
        # 有一列为-1时，对非-1的值进行归一化处理
        valid_ratings = [r for r in ratings if r != -1]
        if len(valid_ratings) > 1:
            min_val = min(valid_ratings)
            max_val = max(valid_ratings)
            
            # 归一化
            normalized = [
                (r - min_val) / (max_val - min_val) if r != max_val and r != -1 else r
                for r in ratings
            ]
        else:
            normalized = ratings  # 只剩一个非-1值，不做变化
    
    # 返回归一化后的数据
    return pd.Series(normalized, index=['StellaRating', 'GteRating', 'DmetaRating'])


def save_weights(ques_info, path):
    # 应用归一化函数到每一行
    ques_info[['StellaRating', 'GteRating', 'DmetaRating']] = ques_info.apply(normalize_ratings, axis=1)

    stella_rating = max(average(list(ques_info['StellaRating'])), 0)  # 防止评分为负
    gte_rating = max(average(list(ques_info['GteRating'])), 0)
    dmeta_rating = max(average(list(ques_info['DmetaRating'])), 0)
    
    sum_rating = stella_rating + gte_rating + dmeta_rating
    
    if sum_rating == 0:
        raise ValueError("sum_rating不能为0")
    # 将权重存储为一个字典
    weights = {
        'LAMBDA_STELLA': round(stella_rating / sum_rating, 4),
        'LAMBDA_GTE': round(gte_rating / sum_rating, 4),
        'LAMBDA_DMETA': round(dmeta_rating / sum_rating, 4)
    }

    print("Stella模型权重:", weights['LAMBDA_STELLA'])
    print("Gte模型权重:", weights['LAMBDA_GTE'])
    print("Dmeta模型权重:", weights['LAMBDA_DMETA'])

    # 保存为 pickle 文件
    with open(path, 'wb') as f:
        pickle.dump(weights, f)

    print("权重已保存到 weights.pkl")


if __name__ == "__main__":
    start = time.time()
    # import_ques()

    session = connect()
    # 读取表trial和user_ques的数据
    trial_info = read_trial(session)
    ques_info = read_ques(session)
    session.close()

    # 对各模型的表现进行评分
    ques_info = rate(ques_info, trial_info)
    save_weights(ques_info, weights_path)
    
    end = time.time()
    print("总用时：" + str(end - start) + 's')
