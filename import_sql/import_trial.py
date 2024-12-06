import time
from utils.db_connect import connect
from sqlalchemy import text
from utils.text_tables import Trial
from utils import trial_path
from read_json import load_json_data
from embedding import encode_ques


def import_trial(data):
    session = connect()

    try:
        if data is not None:
            session.execute(text('TRUNCATE TABLE "Trial" RESTART IDENTITY'))  # 清空表的数据并重置自增序列

            records = []
            for i in range(len(data)):
                trial_record = Trial(
                    ContextId = data.loc[i, 'ContextId'],
                    Context = data.loc[i, 'Context'],
                    Question = data.loc[i, 'Question'],
                    Answers = data.loc[i, 'Answers'],
                    StellaEmbedding = data.loc[i, 'StellaEmbedding'],
                    GteEmbedding = data.loc[i, 'GteEmbedding'],
                    DmetaEmbedding = data.loc[i, 'DmetaEmbedding']
                )
                records.append(trial_record)

            session.bulk_save_objects(records)
            session.commit()
            print("数据写入完成")

    except Exception as e:
        session.rollback()  # 出现异常时回滚
        print(f"Error: {e}")

    finally:
        session.close()


if __name__ == "__main__":
    start = time.time()

    # 读取json文件数据
    data_trial = load_json_data(trial_path)
    # 向量映射
    data_trial = encode_ques(data_trial, 'Question')

    # # 存储数据
    # # import_ques(data_train=data_train)
    # import_ques(data=data_trial)
    end = time.time()
    print("总用时：", str(end - start) + 's')
