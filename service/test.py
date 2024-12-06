import time
from utils.db_connect import connect
from import_sql.import_ques import import_test
from service.read_sql import read_trial, read_test
from service.rerank import *


def test_all(ques_info, trial_info):
    list_stella = list(trial_info['StellaEmbedding'])
    list_gte = list(trial_info['GteEmbedding'])
    list_dmeta = list(trial_info['DmetaEmbedding'])
    
    ques_info['Stella'], ques_info['Gte'], ques_info['Dmeta'], ques_info['Fusion'], ques_info['ReRank'] = 0, 0, 0, 0, 0
    
    for index, row in ques_info.iterrows():
        vec_stella = row['StellaEmbedding']
        vec_gte = row['GteEmbedding']
        vec_dmeta = row['DmetaEmbedding']
        ques = row['Question']
        ideal_ques_id = row['IdealQuestionId']

        score_list = []  # 记录重排模型相似度
        simi_stella, simi_gte, simi_dmeta, simi_fusion = 0, 0, 0, 0  # 初始相似度
        guess_stella, guess_gte, guess_dmeta, guess_fusion = 0, 0, 0, 0  # 每个模型计算下的trial对应Id
        
        item_list = []  # 记录trial中的候选问题id

        for i in range(len(list_stella)):
            guess_id = trial_info.loc[i, 'Id']
            if simi_stella < 1 - cosine(vec_stella, list_stella[i]):
                simi_stella = 1 - cosine(vec_stella, list_stella[i])
                guess_stella = guess_id
                
            if simi_gte < 1 - cosine(vec_gte, list_gte[i]):
                simi_gte = 1 - cosine(vec_gte, list_gte[i])
                guess_gte = guess_id
            
            if simi_dmeta < 1 - cosine(vec_dmeta, list_dmeta[i]):
                simi_dmeta = 1 - cosine(vec_dmeta, list_dmeta[i])
                guess_dmeta = guess_id
            
            
            fusion_stella = (1 - cosine(vec_stella, list_stella[i])) * LAMBDA_STELLA
            fusion_gte = (1 - cosine(vec_gte, list_gte[i])) * LAMBDA_GTE
            fusion_dmeta = (1 - cosine(vec_dmeta, list_dmeta[i])) * LAMBDA_DMETA
            simi_rerank = fusion_stella + fusion_gte + fusion_dmeta
            if simi_fusion < simi_rerank:
                simi_fusion = simi_rerank
                guess_fusion = guess_id
            
            if len(score_list) < NUM_CANDIDATE:
                score_list.append(simi_rerank)
                item_list.append(guess_id)
            else:
                if simi_rerank > min(score_list):
                    item_list[score_list.index(min(score_list))] = guess_id
                    score_list[score_list.index(min(score_list))] = simi_rerank

        ques_categories = classify_ques(ques)  # 问句类型判断

        for i in range(len(item_list)):
            trial_ques = trial_info.loc[trial_info['Id'] == item_list[i], 'Question'].values[0]
            trial_categories = classify_ques(trial_ques)
            if len(ques_categories) == 0:
                if len(trial_categories) != 0:
                    score_list[i] -= CATEGORY_BUFF * (1 - score_list[i])
            elif len(set(ques_categories) & set(trial_categories)) > 0:
                score_list[i] += CATEGORY_BUFF * (1 - score_list[i])
            else:
                score_list[i] -= CATEGORY_BUFF * (1 - score_list[i])

        if guess_stella == ideal_ques_id:
            ques_info.loc[index, 'Stella'] = 1
        else:
            print('Stella错判:', ques)
            
        if guess_gte == ideal_ques_id:
            ques_info.loc[index, 'Gte'] = 1
        else:
            print('Gte错判:', ques)

        if guess_dmeta == ideal_ques_id:
            ques_info.loc[index, 'Dmeta'] = 1
        else:
            print('Dmeta错判:', ques)

        if guess_fusion == ideal_ques_id:
            ques_info.loc[index, 'Fusion'] = 1
        else:
            print('混合模型错判:', ques)

        if ideal_ques_id == item_list[score_list.index(max(score_list))]:
            ques_info.loc[index, 'ReRank'] = 1
        else:
            print('混合+重排错判:', ques)

    print("Stella模型准确率：", round(ques_info['Stella'].mean(), 2))
    print("Gte模型准确率：", round(ques_info['Gte'].mean(), 2))
    print("Dmeta模型准确率：", round(ques_info['Dmeta'].mean(), 2))
    print("混合模型准确率：", round(ques_info['Fusion'].mean(), 2))
    print("混合+重排模型准确率：", round(ques_info['ReRank'].mean(), 2))


if __name__ == "__main__":
    start = time.time()
    import_test()

    session = connect()
    # 读取表trial和user_ques的数据
    trial_info = read_trial(session)
    ques_info = read_test(session)
    session.close()

    test_all(ques_info, trial_info)

    end = time.time()
    print("总用时：" + str(end - start) + 's')
