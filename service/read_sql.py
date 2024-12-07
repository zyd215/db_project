from utils.db_connect import connect
from utils.text_tables import *
import pandas as pd


def read_trial(session=None):
    flag = 0
    if session is None:
        session = connect()
        flag = 1

    query = session.query(
        Trial.Id,
        Trial.Question,
        Trial.StellaEmbedding,
        Trial.GteEmbedding,
        Trial.DmetaEmbedding
    )
    rows = query.all()

    trial_info = pd.DataFrame(rows, columns=['Id', 'Question', 'StellaEmbedding', 'GteEmbedding', 'DmetaEmbedding'])

    if flag:
        session.close()

    return trial_info

def read_ques(session=None):
    flag = 0
    if session is None:
        session = connect()
        flag = 1

    query = session.query(
        UserQuestions.Question,
        UserQuestions.IdealQuestionId,
        UserQuestions.StellaEmbedding,
        UserQuestions.GteEmbedding,
        UserQuestions.DmetaEmbedding
    ).filter(
        UserQuestions.IsDelete == False
    )
    rows = query.all()

    ques_info = pd.DataFrame(rows, columns=['Question', 'IdealQuestionId', 'StellaEmbedding', 'GteEmbedding', 'DmetaEmbedding'])

    if flag:
        session.close()

    return ques_info

def read_test(session=None):
    flag = 0
    if session is None:
        session = connect()
        flag = 1

    query = session.query(
        Test.Question,
        Test.IdealQuestionId,
        Test.StellaEmbedding,
        Test.GteEmbedding,
        Test.DmetaEmbedding
    ).filter(
        Test.IsDelete == False
    )
    rows = query.all()

    ques_info = pd.DataFrame(rows, columns=['Question', 'IdealQuestionId', 'StellaEmbedding', 'GteEmbedding', 'DmetaEmbedding'])

    if flag:
        session.close()

    return ques_info
