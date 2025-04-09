import os
import json
import pandas as pd
from datasets import load_dataset

# path = '/mnt/petrelfs/wangweiyun/workspace_wwy/VLMEvalKit/outputs/InternVL2-8B-MPO-CoT/InternVL2-8B-MPO-CoT_MathVista_MINI_gpt-4-turbo.xlsx'
# path = '/mnt/petrelfs/wangweiyun/workspace_wwy/VLMEvalKit/outputs/InternVL2-8B-MPO-CoT/InternVL2-8B-MPO-CoT_MathVista_TEST_gpt-4-turbo.xlsx'
# path = '/mnt/petrelfs/wangweiyun/workspace_wwy/VLMEvalKit/outputs/InternVL2-5-78B-MPO-try7-1-CoT/InternVL2-5-78B-MPO-try7-1-CoT_MathVista_MINI_gpt-4-turbo.xlsx'
path = '/mnt/petrelfs/wangweiyun/workspace_wwy/VLMEvalKit/outputs/InternVL2-5-8B-MPO-try7-1-CoT/InternVL2-5-8B-MPO-try7-1-CoT_MathVista_MINI_gpt-4-turbo.xlsx'
df = pd.read_excel(path)

if 'MathVista_MINI' in path:
    split = 'testmini'
elif 'MathVista_TEST' in path:
    split = 'test'
else:
    assert False

ds = load_dataset('AI4Math/MathVista', cache_dir=os.path.join('/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev', 'data/MathVista/'))[split]

results = {}
for pred, gold in zip(df.iterrows(), ds):
    row_idx = pred[0]
    
    if split == 'testmini':
        question, choices, answer, question_type, answer_type, index, category, context, skills, source, task, answer_option, prediction, res, log = pred[1]
    else:
        question, choices, answer, question_type, answer_type, index, category, context, skills, source, task, answer_option, image_path, prediction, res, log = pred[1]

    if split == 'testmin':
        assert answer == gold['answer']

    assert index == int(gold['pid']), f"{index=}, {gold['pid']=}"
    assert question_type == gold['question_type']
    assert answer_type == gold['answer_type']

    pid = gold['pid']
    results[pid] = gold.copy()
    results[pid].pop('decoded_image')
    results[pid]['response'] = prediction
    results[pid]['extraction'] = res

assert '.xlsx' in path
save_path = path.replace('.xlsx', '.json')
with open(save_path, 'w') as file:
    json.dump(results, file, indent=4)
