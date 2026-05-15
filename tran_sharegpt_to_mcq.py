"""
ceval_test.json 转换为 ceval_test_val.jsonl
finance_test.json 转换为 finance_test_val.jsonl
mmlu_test.json 转换为 mmlu_test_val.jsonl
原来的格式：ShareGPT 格式，EvalScope 评估的时候会报错，需要转换为 MCQ 格式
"""
import json
import re
import os

def convert_sharegpt_to_mcq(line):
    """将 ShareGPT 格式的一行转换为 MCQ 格式的字典"""
    try:
        data = json.loads(line.strip())
    except:
        return None
    messages = data.get('messages', [])
    if len(messages) < 2:
        return None
    user_msg = messages[0].get('content', '')
    assistant_ans = messages[1].get('content', '').strip()
    
    # 提取问题和选项
    lines = user_msg.strip().split('\n')
    if not lines:
        return None
    question = lines[0].strip()
    options = {}
    # 遍历剩余行，提取选项 A, B, C, D
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # 匹配常见的选项格式：A. xxx  或 A) xxx  或 A  xxx  或 A．xxx  等
        match = re.match(r'^([A-D])[\.\)\s．]+\s*(.*)', line)
        if match:
            opt_key = match.group(1)
            opt_value = match.group(2).strip()
            options[opt_key] = opt_value
    # 补充缺失的选项（如果有的话）
    for letter in ['A','B','C','D']:
        if letter not in options:
            options[letter] = ''
    # 答案提取：假设答案是单个大写字母（可能包含空格或标点）
    answer_match = re.search(r'([A-D])', assistant_ans)
    answer = answer_match.group(1) if answer_match else ''
    return {
        'question': question,
        'A': options['A'],
        'B': options['B'],
        'C': options['C'],
        'D': options['D'],
        'answer': answer
    }

input_files = ['ceval_test.json', 'finance_test.json', 'mmlu_test.json']

def main():
    for infile_name in input_files:
        print("="*50)
        print(infile_name)
        outfile_name = infile_name.replace('.json', '_val.jsonl')
        print(f"正在转换 {infile_name} -> {outfile_name}")
        converted_count = 0
        # 注意：假设当前脚本所在目录与 finetune_data_final 同级，且文件在 finetune_data_final 子目录下
        infile_path = os.path.join('finetune_data_final', infile_name)
        outfile_path = os.path.join('finetune_data_final', outfile_name)
        with open(infile_path, 'r', encoding='utf-8') as infile, \
             open(outfile_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                mcq = convert_sharegpt_to_mcq(line)
                if mcq and mcq['answer']:
                    outfile.write(json.dumps(mcq, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    print(f"警告：{infile_path} 第 {line_num} 行转换失败，已跳过")
        print(f"完成：成功转换 {converted_count} 个样本到 {outfile_path}\n")

if __name__ == '__main__':
    main()