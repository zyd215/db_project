from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.engine import URL


# 测试环境数据库连接信息
url = URL.create(
    drivername="postgresql+psycopg2",
    host="localhost",
    port=5432,
    database="vector",
    username="postgres",
    password=os.environ.get("PASSWORD", "12345678"),  # 换成自己的密码
)

def connect():
    engine = create_engine(url)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()

    return session
