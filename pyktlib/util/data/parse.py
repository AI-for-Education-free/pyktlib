import numpy as np


def check_Q_table(Q_table):
    """
    Q table的每一行和每一列都至少有一个值为1
    :param Q_table:
    :return:
    """
    rows_check = np.any(Q_table == 1, axis=1)
    cols_check = np.any(Q_table == 1, axis=0)
    return np.all(rows_check) and np.all(cols_check)


def parse_Q_table(Q_table):
    """
    根据Q table得出num_question, num_concept, num_max_q2c
    :param Q_table:
    :return:
    """
    num_question, num_concept = Q_table.shape
    num_max_q2c = int(max(Q_table.sum(axis=1)))
    return {
        "num_question": num_question,
        "num_concept": num_concept,
        "num_max_q2c": num_max_q2c
    }


def get_data_statics(data_uniform, num_question):
    """
    根据数据获取统计信息，包括num_seq, num_sample, ave_seq_len, ave_correctness_acc, data_sparsity
    :param data_uniform:
    :param num_question:
    :return:
    """
    num_seq = len(data_uniform)
    num_sample = sum(list(map(lambda x: x["seq_len"], data_uniform)))
    ave_seq_len = round(num_sample/num_seq, 2)
    num_right = 0
    for item_data in data_uniform:
        seq_len = item_data["seq_len"]
        num_right += sum(item_data["correctness_seq"][:seq_len])
    ave_correctness_acc = round(num_right / num_sample, 4)

    U = len(data_uniform)
    Q = num_question
    mat = np.zeros((U, Q))
    for u, item_data in enumerate(data_uniform):
        seq_len = item_data["seq_len"]
        for j in range(seq_len):
            q = item_data["question_seq"][j]
            mat[u][q] = 1
    question_sparsity = round(1 - np.sum(mat) / (U * Q), 4)

    return {
        "num_seq": num_seq,
        "num_sample": num_sample,
        "ave_seq_len": ave_seq_len,
        "ave_correctness_acc": ave_correctness_acc,
        "question_sparsity": question_sparsity
    }
