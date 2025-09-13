import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle, os

df = pd.read_csv('foundation.csv')
# 把每行拼成一句自然语言
df['sentence'] = (
    df['品牌'] + df['产品名称'] + df['色号'] + '，'
    + df['适合肤质'] + '，'
    + df['妆效描述'] + '，遮瑕力' + df['遮瑕力'] + '，价格' + df['价格（元）'].astype(str) + '元。'
)

model = SentenceTransformer('shibing624/text2vec-base-chinese')
embeddings = model.encode(df['sentence'], normalize_embeddings=True)

with open('index.pkl', 'wb') as f:
    pickle.dump({'df': df, 'embeddings': embeddings, 'model_name': 'shibing624/text2vec-base-chinese'}, f)

print('✅ 索引建成，共', len(df), '条')