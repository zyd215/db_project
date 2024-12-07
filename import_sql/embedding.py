import torch
from transformers import BertModel, BertTokenizer
from utils import stella_path, gte_path, dmeta_path


def encode_text(model, tokenizer, text):
    # 对文本进行编码
    encoded_input = tokenizer(text, return_tensors='pt')

    # 通过模型获取文本的向量表示
    with torch.no_grad():
        output = model(**encoded_input)

    # 取输出的最后一层的平均向量
    text_embedding = output.last_hidden_state.mean(dim=1).squeeze()

    return text_embedding

def stella_embedding(text):
    model = BertModel.from_pretrained(stella_path)
    tokenizer = BertTokenizer.from_pretrained(stella_path)

    return encode_text(model, tokenizer, text)

def gte_embedding(text):
    model = BertModel.from_pretrained(gte_path)
    tokenizer = BertTokenizer.from_pretrained(gte_path)

    return encode_text(model, tokenizer, text)

def dmeta_embedding(text):
    model = BertModel.from_pretrained(dmeta_path)
    tokenizer = BertTokenizer.from_pretrained(dmeta_path)

    return encode_text(model, tokenizer, text)

def encode_ques(data, column):
    data['StellaEmbedding'] = data[column].apply(stella_embedding)
    data['GteEmbedding'] = data[column].apply(gte_embedding)
    data['DmetaEmbedding'] = data[column].apply(dmeta_embedding)
    print("数据向量化完成")

    return data
