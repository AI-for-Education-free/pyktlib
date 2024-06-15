import math
import numpy as np
import pandas as pd

from ..util.data.load_file import load_csv
from .util import map_user_info


def preprocess_assist2009(data_path):
    useful_cols = [
        "order_id", "user_id", "problem_id", "correct", "skill_id", "school_id", "skill_name", 'attempt_count',
        'hint_count', "ms_first_response", "overlap_time"
    ]
    cols_rename_map = {
        "problem_id": "question_id",
        "skill_id": "concept_id",
        "correct": "correctness",
        "skill_name": "concept_name",
        "attempt_count": "num_attempt",
        "hint_count": "num_hint",
        "ms_first_response": "use_time_first_attempt",
        "overlap_time": "use_time"
    }
    df = load_csv(data_path, useful_cols, cols_rename_map)
    # 有知识点名称的interaction数量为325637，总数量为401756
    df.dropna(subset=["question_id", "concept_id"], inplace=True)
    df["question_id"] = df["question_id"].map(int)
    df["concept_id"] = df["concept_id"].map(int)
    # 该数据集use_time_first_attempt, num_hint, num_attempt都没有nan，
    # 有4684条数据use_time_first_attempt <= 0，
    # num attempt和num hint都>=0
    df["use_time_first_attempt"] = df["use_time_first_attempt"].map(
        lambda t: max(1, math.ceil(t / 1000))
    )
    df["use_time"] = df["use_time"].map(lambda t: max(1, math.ceil(t / 1000)))
    # 关于num attempt和num hint，有脏数据，如attempt的数量大于100，或者为0，但是这里不处理，因为不同论文处理方式不一样，这里给出原始数据
    # 官网对attempt count的定义是Number of student attempts on this problem，
    # 没说是从第一次做开始就计数，还是做错了一次后开始计数，我们按照LBKT的设定，假设可以为0）

    # 知识点和习题id重映射
    concept_ids = sorted(list(pd.unique(df["concept_id"])))
    question_ids = sorted(list(pd.unique(df["question_id"])))
    question_id_map = {q_id_original: q_id_mapped for q_id_mapped, q_id_original in enumerate(question_ids)}
    concept_id_map = {c_id_original: c_id_mapped for c_id_mapped, c_id_original in enumerate(concept_ids)}
    df["question_id"] = df["question_id"].map(question_id_map)
    df["concept_id"] = df["concept_id"].map(concept_id_map)

    # 获取concept name和id的对应
    concept_names = list(pd.unique(df.dropna(subset=["concept_name"])["concept_name"]))
    concept_id2name = {}
    for c_name in concept_names:
        concept_data = df[df["concept_name"] == c_name]
        c_id = int(concept_data["concept_id"].iloc[0])
        concept_id2name[c_id] = c_name.strip()
    concept_id2name_map = pd.DataFrame({
        "concept_id": concept_id2name.keys(),
        "concept_name": concept_id2name.values()
    })

    # 为了方便后续认知诊断任务，重映射user id
    user_ids = sorted(list(pd.unique(df["user_id"])))
    user_id_map = {u_id_original: u_id_mapped for u_id_mapped, u_id_original in enumerate(user_ids)}
    df["user_id"] = df["user_id"].map(user_id_map)

    # school_id按照学生数量重映射
    df["school_id"] = df["school_id"].fillna(-1)
    df["school_id"] = df["school_id"].map(int)
    school_id_map, school_info = map_user_info(df, "school_id")

    df_qc = pd.DataFrame({
        "question_id": map(int, df["question_id"].tolist()),
        "concept_id": map(int, df["concept_id"].tolist())
    })
    question_concept_map = {}
    for question_id, group_info in df_qc[["question_id", "concept_id"]].groupby("question_id"):
        c_ids = sorted(pd.unique(group_info["concept_id"]).tolist())
        question_concept_map.setdefault(question_id, c_ids)
    Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
    for q_id in question_concept_map.keys():
        correspond_c_ids = question_concept_map[q_id]
        Q_table[[q_id] * len(correspond_c_ids), correspond_c_ids] = [1] * len(correspond_c_ids)

    # 去除多知识点习题的冗余
    df = df[~df.duplicated(subset=["user_id", "order_id", "question_id"])]

    # 序列元素和原始df field之间的对应关系
    seq_item_map = {
        "question_seq": "question_id",
        "correctness_seq": "correctness",
        "use_time_seq": "use_time",
        "use_time_first_seq": "use_time_first_attempt",
        "num_hint_seq": "num_hint",
        "num_attempt_seq": "num_attempt"
    }
    id_keys = list(set(df.columns) - set(seq_item_map.values()) - {"order_id", "concept_name", "concept_id"})
    seq_keys = [
        "question_seq", "correctness_seq", "use_time_seq", "use_time_first_seq", "num_hint_seq", "num_attempt_seq"
    ]

    data_uniform = []
    for user_id in pd.unique(df["user_id"]):
        user_data = df[df["user_id"] == user_id]
        user_data = user_data.sort_values(by=["order_id"])
        if len(user_data) < 2:
            continue
        object_data = {seq_key: [] for seq_key in seq_keys}
        for k in id_keys:
            object_data[k] = user_data.iloc[0][k]
        for i, (_, row_data) in enumerate(user_data.iterrows()):
            for seq_key in seq_keys:
                object_data[seq_key].append(row_data[seq_item_map[seq_key]])
        object_data["seq_len"] = len(object_data["correctness_seq"])
        data_uniform.append(object_data)

    return data_uniform, Q_table
