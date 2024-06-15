import os
import numpy as np

from ..util.data.load_file import load_csv, load_json


def preprocess_xes3g5m(data_dir):
    # kc_level和question_level的数据是一样的，前者是multi_concept，后者是only_question（对于多知识点习题用 _ 拼接知识点）
    train_valid_path = os.path.join(data_dir, "question_level", "train_valid_sequences_quelevel.csv")
    test_path = os.path.join(data_dir, "question_level", "test_quelevel.csv")
    df_train_valid = load_csv(
        train_valid_path, ["uid", "questions", "concepts", "responses", "timestamps", "selectmasks"]
    )
    df_test = load_csv(test_path, ["uid", "questions", "concepts", "responses", "timestamps"])

    # metadata
    question_meta_path = os.path.join(data_dir, "metadata", "questions.json")
    concept_meta_path = os.path.join(data_dir, "metadata", "kc_routes_map.json")
    question_meta = load_json(question_meta_path)
    concept_meta = load_json(concept_meta_path)

    concept_meta = {int(c_id): c_name.strip() for c_id, c_name in concept_meta.items()}
    question_meta = {int(q_id): q_meta for q_id, q_meta in question_meta.items()}

    data_all = {}
    for i in df_train_valid.index:
        user_id = int(df_train_valid["uid"][i])
        data_all.setdefault(user_id, {
            "question_seq": [],
            "concept_seq": [],
            "correctness_seq": [],
            "time_seq": []
        })
        # df_train_valid提供的数据是切割好的（将长序列切成固定长度为200的序列），不足200的用-1补齐
        mask_seq = list(map(int, df_train_valid["selectmasks"][i].split(",")))
        if -1 in mask_seq:
            end_pos = mask_seq.index(-1)
        else:
            end_pos = 200

        question_seq = list(map(int, df_train_valid["questions"][i].split(",")))[:end_pos]
        concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                               df_train_valid["concepts"][i].split(",")))[:end_pos]
        correctness_seq = list(map(int, df_train_valid["responses"][i].split(",")))[:end_pos]
        time_seq = list(map(int, df_train_valid["timestamps"][i].split(",")))[:end_pos]
        data_all[user_id]["question_seq"] += question_seq
        data_all[user_id]["concept_seq"] += concept_seq
        data_all[user_id]["correctness_seq"] += correctness_seq
        data_all[user_id]["time_seq"] += time_seq

    for i in df_test.index:
        # df_test提供的数据是未切割的
        user_id = int(df_test["uid"][i])
        data_all.setdefault(user_id, {
            "question_seq": [],
            "concept_seq": [],
            "correctness_seq": [],
            "time_seq": []
        })
        question_seq = list(map(int, df_test["questions"][i].split(",")))
        concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                               df_test["concepts"][i].split(",")))
        correctness_seq = list(map(int, df_test["responses"][i].split(",")))
        time_seq = list(map(int, df_test["timestamps"][i].split(",")))
        data_all[user_id]["question_seq"] += question_seq
        data_all[user_id]["concept_seq"] += concept_seq
        data_all[user_id]["correctness_seq"] += correctness_seq
        data_all[user_id]["time_seq"] += time_seq

    data_uniform = [{
        "user_id": user_id,
        "question_seq": seqs["question_seq"],
        "concept_seq": seqs["concept_seq"],
        "correctness_seq": seqs["correctness_seq"],
        "time_seq": seqs["time_seq"],
        "seq_len": len(seqs["correctness_seq"])
    } for user_id, seqs in data_all.items()]

    # 提取每道习题对应的知识点：提供的数据（train_valid_sequences_quelevel.csv和test_quelevel.csv）中习题对应的知识点是最细粒度的，类似edi2020数据集中层级知识点里最细粒度的知识点
    # 而question metadata里每道题的kc routes是完整的知识点（层级）
    # 并且提供的数据中习题对应知识点和question metadata中习题对应的知识点不是完全一一对应的，例如习题1035
    # 在question metadata中对应的知识点为
    # ['拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄差',
    #  '能力----运算求解',
    #  '课内题型----综合与实践----应用题----倍数问题----已知两量之间倍数关系和两量之差，求两个量',
    #  '学习能力----七大能力----运算求解',
    #  '拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄问题基本关系和差问题',
    #  '课内知识点----数与运算----数的运算的实际应用（应用题）----整数的简单实际问题----除法的实际应用',
    #  '知识点----应用题----和差倍应用题----已知两量之间倍数关系和两量之差，求两个量',
    #  '知识点----数的运算----估算与简单应用----整数的简单实际问题----除法的实际应用']
    # 在数据中对应的知识点为[169, 177, 239, 200, 73]，其对应的知识点名称为['除法的实际应用', '已知两量之间倍数关系和两量之差，求两个量', '年龄差', '年龄问题基本关系和差问题', '运算求解']
    # 选择使用数据的对应关系（官方数据已经帮忙在最细粒度上知识点去重过了）

    question_ids = []
    concept_ids = []
    question_concept_map = {}
    for item_data in data_uniform:
        for i in range(item_data["seq_len"]):
            question_ids.append(item_data["question_seq"][i])
            concept_ids.extend(item_data["concept_seq"][i])

    # 习题和知识点id都是映射过的，但是习题共有7651个，其id却是从0开始，7651结束（有一个空缺），为了不影响后续模型训练，再映射一次习题id
    question_ids = sorted(list(set(question_ids)))
    question_id_map = {q_id_original: q_id_mapped for q_id_mapped, q_id_original in enumerate(question_ids)}
    for item_data in data_uniform:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            c_ids = item_data["concept_seq"][i]
            item_data["question_seq"][i] = question_id_map[q_id]
            question_concept_map.setdefault(question_id_map[q_id], c_ids)

    concept_ids = sorted(list(set(concept_ids)))
    Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
    for q_id in question_concept_map.keys():
        correspond_c_ids = question_concept_map[q_id]
        Q_table[[q_id] * len(correspond_c_ids), correspond_c_ids] = [1] * len(correspond_c_ids)

    for item_data in data_uniform:
        del item_data["concept_seq"]

    # 为了方便后续认知诊断任务，重映射user id
    user_ids = sorted(list(set([item_data["user_id"] for item_data in data_uniform])))
    user_id_map = {u_id_original: u_id_mapped for u_id_mapped, u_id_original in enumerate(user_ids)}
    for item_data in data_uniform:
        item_data["user_id"] = user_id_map[item_data["user_id"]]

    return data_uniform, Q_table
