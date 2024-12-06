import time
from utils.db_connect import connect
from sqlalchemy import text
from utils.text_tables import UserQuestions, Test
from import_sql.read_xlsx import load_excel_data
from import_sql.embedding import encode_ques
from utils import ques_path, test_path


def import_ques(path=ques_path):
    start = time.time()

    data = load_excel_data(path)
    data = encode_ques(data, 'Question')  # 向量映射

    session = connect()
    try:
        if data is not None:
            session.execute(text('TRUNCATE TABLE "UserQuestions" RESTART IDENTITY'))  # 清空表的数据并重置自增序列

            records = []
            for i in range(len(data)):
                ques_record = UserQuestions(
                    Question = data.loc[i, 'Question'],
                    IdealQuestionId =  int(data.loc[i, 'IdealQuestionId']),
                    StellaEmbedding = data.loc[i, 'StellaEmbedding'],
                    GteEmbedding = data.loc[i, 'GteEmbedding'],
                    DmetaEmbedding = data.loc[i, 'DmetaEmbedding'],
                    IsDelete = data.loc[i, 'IsDelete']
                )
                records.append(ques_record)

            session.bulk_save_objects(records)
            session.commit()
            print("数据写入完成")

    except Exception as e:
        session.rollback()  # 出现异常时回滚
        print(f"Error: {e}")

    finally:
        session.close()
        end = time.time()
        print("用户问题写入用时：", str(end - start) + 's')


def import_test(path=test_path):
    start = time.time()

    data = load_excel_data(path)
    data = encode_ques(data, 'Question')  # 向量映射

    session = connect()
    try:
        if data is not None:
            session.execute(text('TRUNCATE TABLE "Test" RESTART IDENTITY'))  # 清空表的数据并重置自增序列

            records = []
            for i in range(len(data)):
                ques_record = Test(
                    Question = data.loc[i, 'Question'],
                    IdealQuestionId =  int(data.loc[i, 'IdealQuestionId']),
                    StellaEmbedding = data.loc[i, 'StellaEmbedding'],
                    GteEmbedding = data.loc[i, 'GteEmbedding'],
                    DmetaEmbedding = data.loc[i, 'DmetaEmbedding'],
                    IsDelete = data.loc[i, 'IsDelete']
                )
                records.append(ques_record)

            session.bulk_save_objects(records)
            session.commit()
            print("数据写入完成")

    except Exception as e:
        session.rollback()  # 出现异常时回滚
        print(f"Error: {e}")

    finally:
        session.close()
        end = time.time()
        print("用户问题写入用时：", str(end - start) + 's')


if __name__ == "__main__":
    import_ques()
    import_test()
