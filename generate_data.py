import json
import pandas as pd
import random
import os

random.seed(42)


def load_mmlu_parquet(filepath):
    df = pd.read_parquet(filepath)
    items = []
    for _, row in df.iterrows():
        question = str(row["question"])
        choices = row["choices"]
        answer_idx = row["answer"]
        if len(choices) >= 4:
            item = {
                "question": question,
                "A": str(choices[0]),
                "B": str(choices[1]),
                "C": str(choices[2]),
                "D": str(choices[3]),
                "answer": chr(ord("A") + answer_idx),
            }
            items.append(item)
    return items


def load_ceval_parquet(filepath):
    df = pd.read_parquet(filepath)
    items = []
    for _, row in df.iterrows():
        answer = str(row["answer"]).strip().upper()
        if answer not in ["A", "B", "C", "D"]:
            continue
        item = {
            "question": str(row["question"]),
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
            "D": str(row["D"]),
            "answer": answer,
        }
        items.append(item)
    return items


def load_csv_to_mcq(filepath):
    df = pd.read_csv(filepath, encoding="utf-8")
    items = []
    for _, row in df.iterrows():
        item = {
            "question": str(row["question"]),
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
        }
        if "D" in df.columns:
            item["D"] = str(row["D"])
        else:
            item["D"] = ""
        answer = ""
        if "answer" in df.columns:
            answer = str(row["answer"])
        elif "Answer" in df.columns:
            answer = str(row["Answer"])
        if answer in ["nan", "NaN", "NAN", "", "None", "none"]:
            continue
        if len(answer) > 1:
            continue
        item["answer"] = answer
        if not item["D"]:
            continue
        items.append(item)
    return items


def mcq_to_messages(item):
    question = item["question"]
    a = item["A"]
    b = item["B"]
    c = item["C"]
    d = item["D"]
    content = question + "\nA. " + a + "\nB. " + b + "\nC. " + c + "\nD. " + d
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": item["answer"]},
        ]
    }


def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "finetune_data_final")

    # ========== 1. 加载金融数据 ==========
    finance_data = []

    ant_finance_dir = os.path.join(
        base_dir, "financial_evaluation_dataset", "data", "Ant", "金融知识"
    )
    if os.path.exists(ant_finance_dir):
        ant_files = [
            f
            for f in os.listdir(ant_finance_dir)
            if f.endswith(".csv")
            and "执业医师" not in f
            and "执业药师" not in f
        ]
        for f in ant_files:
            filepath = os.path.join(ant_finance_dir, f)
            items = load_csv_to_mcq(filepath)
            finance_data.extend(items)
            print(f"加载 Ant/{f}: {len(items)} 条")

    sufe_dir = os.path.join(
        base_dir, "financial_evaluation_dataset", "data", "SUFE"
    )
    if os.path.exists(sufe_dir):
        for split in ["dev", "val"]:
            split_dir = os.path.join(sufe_dir, split)
            if not os.path.exists(split_dir):
                continue
            files = [f for f in os.listdir(split_dir) if f.endswith(".csv")]
            for f in files:
                filepath = os.path.join(split_dir, f)
                items = load_csv_to_mcq(filepath)
                finance_data.extend(items)
        print(f"加载 SUFE (dev+val): 完成")

    random.shuffle(finance_data)
    finance_data = finance_data * 2
    random.shuffle(finance_data)
    print(f"金融数据(增强后): {len(finance_data)} 条")

    # ========== 2. 加载 MMLU 数据 ==========
    mmlu_dev = []
    mmlu_test_all = []
    mmlu_dir = os.path.join(base_dir, "mmlu")
    if os.path.exists(mmlu_dir):
        mmlu_subjects = [
            d
            for d in os.listdir(mmlu_dir)
            if os.path.isdir(os.path.join(mmlu_dir, d))
            and not d.startswith(".")
            and d != ".cache"
        ]
        for subject in mmlu_subjects:
            dev_file = os.path.join(mmlu_dir, subject, "dev-00000-of-00001.parquet")
            if os.path.exists(dev_file):
                try:
                    items = load_mmlu_parquet(dev_file)
                    mmlu_dev.extend(items)
                except Exception:
                    pass
            test_file = os.path.join(mmlu_dir, subject, "test-00000-of-00001.parquet")
            if os.path.exists(test_file):
                try:
                    items = load_mmlu_parquet(test_file)
                    mmlu_test_all.extend(items)
                except Exception:
                    pass
    print(f"MMLU dev: {len(mmlu_dev)} 条, test: {len(mmlu_test_all)} 条")

    # ========== 3. 加载 C-Eval 数据 ==========
    ceval_dev = []
    ceval_test_all = []
    ceval_dir = os.path.join(base_dir, "ceval")
    if os.path.exists(ceval_dir):
        ceval_subjects = [
            d
            for d in os.listdir(ceval_dir)
            if os.path.isdir(os.path.join(ceval_dir, d)) and not d.startswith(".")
        ]
        for subject in ceval_subjects:
            dev_file = os.path.join(ceval_dir, subject, "dev-00000-of-00001.parquet")
            if os.path.exists(dev_file):
                items = load_ceval_parquet(dev_file)
                ceval_dev.extend(items)
            test_file = os.path.join(ceval_dir, subject, "test-00000-of-00001.parquet")
            if os.path.exists(test_file):
                items = load_ceval_parquet(test_file)
                ceval_test_all.extend(items)
    print(f"C-Eval dev: {len(ceval_dev)} 条, test: {len(ceval_test_all)} 条")

    # ========== 4. 采样训练集 ==========
    general_data = mmlu_dev + ceval_dev
    random.shuffle(general_data)

    train_size = 2000
    train_finance_count = int(train_size * 0.8)
    train_general_count = train_size - train_finance_count

    train_finance = finance_data[:train_finance_count]
    train_general = general_data[:train_general_count]

    train_data = [mcq_to_messages(item) for item in train_finance + train_general]
    random.shuffle(train_data)

    # ========== 5. 采样测试集 ==========
    random.shuffle(mmlu_test_all)
    mmlu_test = random.sample(mmlu_test_all, min(500, len(mmlu_test_all)))
    mmlu_test_data = [mcq_to_messages(item) for item in mmlu_test]

    random.shuffle(ceval_test_all)
    ceval_test = random.sample(ceval_test_all, min(500, len(ceval_test_all)))
    ceval_test_data = [mcq_to_messages(item) for item in ceval_test]

    remaining_finance = finance_data[train_finance_count:]
    finance_test = random.sample(remaining_finance, min(1000, len(remaining_finance)))
    finance_test_data = [mcq_to_messages(item) for item in finance_test]

    # ========== 6. 保存文件 ==========
    os.makedirs(output_dir, exist_ok=True)

    save_jsonl(train_data, os.path.join(output_dir, "train.json"))
    save_jsonl(mmlu_test_data, os.path.join(output_dir, "mmlu_test.json"))
    save_jsonl(ceval_test_data, os.path.join(output_dir, "ceval_test.json"))
    save_jsonl(finance_test_data, os.path.join(output_dir, "finance_test.json"))

    dataset_info = {
        "finetune_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"\n数据集准备完成:")
    print(f"  训练集: {len(train_data)} 条 (金融 {len(train_finance)} + 通用 {len(train_general)})")
    print(f"  MMLU 测试集: {len(mmlu_test_data)} 条")
    print(f"  C-Eval 测试集: {len(ceval_test_data)} 条")
    print(f"  Fin-Eva 测试集: {len(finance_test_data)} 条")
    print(f"\n文件路径: {output_dir}")


if __name__ == "__main__":
    main()
