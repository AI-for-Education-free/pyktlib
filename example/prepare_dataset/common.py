import argparse
import os

from pyktlib.util.data.basic import read_kt_data, write_kt_data
from pyktlib.data_preprocess.prepare_dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_data_path", type=str,
                        default=r"F:\code\myProjects\pyktlib\example\data\preprocessed\xes3g5m\data.txt")
    parser.add_argument("--save_dir", type=str,
                        default=r"F:\code\myProjects\pyktlib\example\data\settings\common")
    parser.add_argument("--save_name", type=str, default="xes3g5m")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_radio", type=float, default=0.2)
    args = parser.parse_args()
    params = vars(args)

    data = read_kt_data(params["preprocessed_data_path"])
    dataset = truncate2multi_seq(data, max_seq_len=params["max_seq_len"])
    dataset = list(filter(lambda item_data: item_data["seq_len"] >= params["min_seq_len"], dataset))
    datasets_train, datasets_valid, dataset_test = five_fold_spilt1(dataset, params["test_radio"], params["seed"])
    write_kt_data(dataset_test, os.path.join(params["save_dir"], f"{params['save_name']}_test.txt"))
    for fold in range(5):
        dataset_train = datasets_train[fold]
        dataset_valid = datasets_valid[fold]
        write_kt_data(dataset_train, os.path.join(params["save_dir"], f"{params['save_name']}_train_fold_{fold}.txt"))
        write_kt_data(dataset_valid, os.path.join(params["save_dir"], f"{params['save_name']}_valid_fold_{fold}.txt"))
