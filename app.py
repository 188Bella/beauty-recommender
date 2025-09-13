import pickle, gradio as gr, torch
from sentence_transformers import SentenceTransformer
print("⏳ 正在加载模型 'all-MiniLM-L6-v2'...") 

with open('index.pkl', 'rb') as f:
    bundle = pickle.load(f)
df, embeddings, model_name = bundle['df'], bundle['embeddings'], bundle['model_name']
model = SentenceTransformer(model_name)

def recommend(skintype, budget, tone):
    user_txt = f'{skintype}，预算{budget}，色调{tone}。'
    user_vec = model.encode(user_txt, normalize_embeddings=True)
    scores = torch.tensor(user_vec @ embeddings.T)
    top_idx = scores.argmax().item()
    row = df.iloc[top_idx]
    return (f"推荐：{row['品牌']}{row['产品名称']} {row['色号']}\n"
            f"适合：{row['适合肤质']} | 遮瑕：{row['遮瑕力']} | 价格：{row['价格（元）']}元\n"
            f"妆效：{row['妆效描述']}")

demo = gr.Interface(
    fn=recommend,
    inputs=[gr.Dropdown(['油皮','干皮','混干','混油','中性皮肤','敏感肌'], label='肤质'),
            gr.Dropdown(['低（<150）','中（150-300）','高（>300）'], label='预算'),
            gr.Dropdown(['冷调','暖调','中性'], label='色调')],
    outputs=gr.Textbox(label='推荐结果'),
    title='粉底液小助手 · Mini版',
    description='选 3 个标签，立刻给你最相似的产品！'
)
demo.launch(share=True)   # 自动生成 https 外链