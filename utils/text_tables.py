import sqlalchemy  # 防止指标名称和变量类型名称相同
from sqlalchemy import Column
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Trial(Base):
    __tablename__ = 'Trial'

    Id = Column(sqlalchemy.Integer, primary_key=True)
    ContextId = Column(sqlalchemy.Text)
    Context = Column(sqlalchemy.Text)
    Question = Column(sqlalchemy.Text)
    Answers = Column(sqlalchemy.Text)
    StellaEmbedding = Column(Vector(768))
    GteEmbedding = Column(Vector(768))
    DmetaEmbedding = Column(Vector(768))

class UserQuestions(Base):
    __tablename__ = 'UserQuestions'

    Id = Column(sqlalchemy.Integer, primary_key=True)
    Question = Column(sqlalchemy.Text)
    IdealQuestionId = Column(sqlalchemy.Integer)
    StellaEmbedding = Column(Vector(768))
    GteEmbedding = Column(Vector(768))
    DmetaEmbedding = Column(Vector(768))
    IsDelete = Column(sqlalchemy.Boolean)

class Test(Base):
    __tablename__ = 'Test'

    Id = Column(sqlalchemy.Integer, primary_key=True)
    Question = Column(sqlalchemy.Text)
    IdealQuestionId = Column(sqlalchemy.Integer)
    StellaEmbedding = Column(Vector(768))
    GteEmbedding = Column(Vector(768))
    DmetaEmbedding = Column(Vector(768))
    IsDelete = Column(sqlalchemy.Boolean)
    