import random


def truncate2multi_seq(data_uniform, max_seq_len=200):
    """
    将一个用户的数据进行截断多段
    :param data_uniform:
    :param max_seq_len:
    :return:
    """
    id_keys = []
    seq_keys = []
    for k, v in data_uniform[0].items():
        if type(v) is list:
            seq_keys.append(k)
        else:
            id_keys.append(k)

    result = []
    for item_data in data_uniform:
        seq_len = item_data["seq_len"]
        if seq_len <= max_seq_len:
            item_data_new = {key: item_data[key] for key in id_keys}
            pad_len = max_seq_len - seq_len
            for k in seq_keys:
                item_data_new[k] = item_data[k][0:seq_len] + [0] * pad_len
            item_data_new["mask_seq"] = [1] * seq_len + [0] * pad_len
            result.append(item_data_new)
        else:
            num_segment = item_data["seq_len"] // max_seq_len
            num_segment = num_segment if (item_data["seq_len"] % max_seq_len == 0) else (num_segment + 1)
            for segment in range(num_segment):
                item_data_new = {key: item_data[key] for key in id_keys}
                start_index = max_seq_len * segment
                if segment == item_data["seq_len"] // max_seq_len:
                    # the last segment
                    pad_len = max_seq_len - (item_data["seq_len"] % max_seq_len)
                    for k in seq_keys:
                        item_data_new[k] = item_data[k][start_index:] + [0] * pad_len
                    item_data_new["seq_len"] = item_data["seq_len"] % max_seq_len
                    item_data_new["mask_seq"] = [1] * (max_seq_len - pad_len) + [0] * pad_len
                else:
                    end_index = max_seq_len * (segment + 1)
                    for k in seq_keys:
                        item_data_new[k] = item_data[k][start_index:end_index]
                    item_data_new["seq_len"] = max_seq_len
                    item_data_new["mask_seq"] = [1] * max_seq_len
                result.append(item_data_new)

    return result


def truncate2one_seq(data_uniform, max_seq_len=200, from_start=True):
    """
    截断数据，取最前面或者最后面一段
    :param data_uniform:
    :param from_start:
    :param max_seq_len:
    :return:
    """
    id_keys = []
    seq_keys = []
    for k, v in data_uniform[0].items():
        if type(v) is list:
            seq_keys.append(k)
        else:
            id_keys.append(k)

    result = []
    for item_data in data_uniform:
        item_data_new = {key: item_data[key] for key in id_keys}
        seq_len = item_data["seq_len"]
        start_index, end_index = 0, seq_len
        if seq_len > max_seq_len and from_start:
            end_index = max_seq_len
        if seq_len > max_seq_len and not from_start:
            start_index = end_index - max_seq_len
        pad_len = max_seq_len - end_index + start_index
        for k in seq_keys:
            item_data_new[k] = item_data[k][start_index:end_index] + [0] * pad_len
        item_data_new["seq_len"] = end_index - start_index
        item_data_new["mask_seq"] = [1] * item_data_new["seq_len"] + \
                                    [0] * (max_seq_len - item_data_new["seq_len"])
        result.append(item_data_new)

    return result


def five_fold_spilt1(dataset_uniformed, test_radio, seed=0):
    """
    选一部分数据做测试集，剩余数据用n折交叉划分为训练集和验证集
    :param test_radio:
    :param dataset_uniformed:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_4], [valid_fold_0, ..., valid_fold_4], test)
    """
    random.seed(seed)
    random.shuffle(dataset_uniformed)
    n_fold = 5

    num_all = len(dataset_uniformed)
    num_train_valid = int(num_all * (1 - test_radio))
    num_fold = (num_train_valid // n_fold) + 1

    dataset_test = dataset_uniformed[num_train_valid:]
    dataset_train_valid = dataset_uniformed[:num_train_valid]
    dataset_folds = [dataset_train_valid[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], dataset_test)
    for i in range(n_fold):
        fold_valid = i
        result[1].append(dataset_folds[fold_valid])
        folds_train = set(range(n_fold)) - {fold_valid}
        data_train = []
        for fold in folds_train:
            data_train += dataset_folds[fold]
        result[0].append(data_train)

    return result


def five_fold_spilt2(dataset_uniformed, valid_radio, seed=0):
    """
    先用n折交叉划分为训练集和测试集，再在训练集中划分一部分数据为验证集
    :param valid_radio:
    :param dataset_uniformed:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_4], [valid_fold_0, ..., valid_fold_4], [test_fold_0, ..., test_fold_4])
    """
    random.seed(seed)
    random.shuffle(dataset_uniformed)
    n_fold = 5

    num_all = len(dataset_uniformed)
    num_fold = (num_all // n_fold) + 1

    dataset_folds = [dataset_uniformed[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], [])
    for i in range(n_fold):
        fold_test = i
        result[2].append(dataset_folds[fold_test])
        folds_train_valid = set(range(n_fold)) - {fold_test}
        dataset_train_valid = []
        for fold in folds_train_valid:
            dataset_train_valid += dataset_folds[fold]
        random.shuffle(dataset_train_valid)
        num_valid = int(len(dataset_train_valid) * valid_radio)
        result[1].append(dataset_train_valid[:num_valid])
        result[0].append(dataset_train_valid[num_valid:])

    return result
