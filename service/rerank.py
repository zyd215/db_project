import time
import numpy as np
import pickle
from scipy.spatial.distance import cosine
import jieba.posseg as pseg
from utils.db_connect import connect
from service.read_sql import *
from service import *


# 从 pickle 文件加载权重
with open(weights_path, 'rb') as f:
    weights = pickle.load(f)

# 获取权重值
LAMBDA_STELLA = weights['LAMBDA_STELLA']
LAMBDA_GTE = weights['LAMBDA_GTE']
LAMBDA_DMETA = weights['LAMBDA_DMETA']

NUM_CANDIDATE = 3  # 候选问题数量
CATEGORY_BUFF = 0.4  # 重排参数

def classify_ques(ques, print_out=False):
    result = pseg.cut(ques)
    allowed_pos = {"n", "r", "t", "m", "v", 'vn'}  # 过滤词性
    filtered_words = [word for word, flag in result if flag in allowed_pos]
    ques_categories = []
    for word in filtered_words:
        if print_out:
            print(word, end=" ")
        if word in word_category.keys():
            ques_categories.append(word_category[word])

    return ques_categories

# 各模型评价
def re_rank(ques_info, trial_info, buff=CATEGORY_BUFF):
    list_stella = list(trial_info['StellaEmbedding'])
    list_gte = list(trial_info['GteEmbedding'])
    list_dmeta = list(trial_info['DmetaEmbedding'])

    def rank(row):
        vec_stella = row['StellaEmbedding']
        vec_gte = row['GteEmbedding']
        vec_dmeta = row['DmetaEmbedding']
        ques = row['Question']
        ideal_ques_id = row['IdealQuestionId']

        score_list = []  # 记录相似度
        item_list = []  # 记录trial中的候选问题id

        for i in range(len(list_stella)):
            simi_stella = (1 - cosine(vec_stella, list_stella[i])) * LAMBDA_STELLA
            simi_gte = (1 - cosine(vec_gte, list_gte[i])) * LAMBDA_GTE
            simi_dmeta = (1 - cosine(vec_dmeta, list_dmeta[i])) * LAMBDA_DMETA
            similarity = simi_stella + simi_gte + simi_dmeta

            if len(score_list) < NUM_CANDIDATE:
                score_list.append(similarity)
                item_list.append(trial_info.loc[i, 'Id'])
            else:
                if similarity > min(score_list):
                    item_list[score_list.index(min(score_list))] = trial_info.loc[i, 'Id']
                    score_list[score_list.index(min(score_list))] = similarity

        ques_categories = classify_ques(ques)  # 问句类型判断

        for i in range(len(item_list)):
            trial_ques = trial_info.loc[trial_info['Id'] == item_list[i], 'Question'].values[0]
            trial_categories = classify_ques(trial_ques)
            if len(ques_categories) == 0:
                if len(trial_categories) != 0:
                    score_list[i] -= buff * (1 - score_list[i])
            elif len(set(ques_categories) & set(trial_categories)) > 0:
                score_list[i] += buff * (1 - score_list[i])
            else:
                score_list[i] -= buff * (1 - score_list[i])

        if ideal_ques_id == item_list[score_list.index(max(score_list))]:
            return 1
        
        guess_ques = trial_info.loc[trial_info['Id'] == item_list[score_list.index(max(score_list))], 'Question'].values[0]
        ideal_ques = trial_info.loc[trial_info['Id'] == ideal_ques_id, 'Question'].values[0]
        print("用户问题："+ques+" 误判问题："+guess_ques+" 真实问题："+ideal_ques)
        print(score_list)
        return 0

    ques_info['IsRight'] = ques_info.apply(rank, axis=1)
    return ques_info


if __name__ == "__main__":
    start = time.time()

    session = connect()
    # 读取表trial和user_ques的数据
    trial_info = read_trial(session)
    ques_info = read_ques(session)
    # ques_info = read_test(session)
    session.close()

    # # 对各模型的表现进行评分
    # ques_info = re_rank(ques_info, trial_info)
    # print(ques_info['IsRight'])

    # end = time.time()
    # print("总用时：" + str(end - start) + 's')

    import matplotlib.pyplot as plt
    ratio_list = []
    buff_list = list(np.arange(0, 0.8, 0.02))
    for buff in buff_list:
        # 对各模型的表现进行评分
        ques_info = re_rank(ques_info, trial_info, buff)
        ratio_ones = ques_info['IsRight'].mean()
        if ratio_ones < 1:
            print(buff)
        ratio_list.append(ratio_ones)
        
    # 创建一个图形
    plt.figure(figsize=(8, 6))

    # 绘制折线图
    plt.plot(buff_list, ratio_list, color='b')

    # 添加网格
    plt.grid(True)

    # 设置标签和标题
    plt.xlabel('buff')
    plt.ylabel('Accuracy')

    # 显示图形
    plt.savefig(accuracy_path)
    
    end = time.time()
    print("总用时：" + str(end - start) + 's')
    