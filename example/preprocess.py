import argparse
import os
import numpy as np

from pyktlib.data_preprocess.assist2009 import preprocess_assist2009
from pyktlib.data_preprocess.xes3g5m import preprocess_xes3g5m
from pyktlib.util.data.basic import write_kt_data, write_json
from pyktlib.util.data.parse import parse_Q_table, get_data_statics

preprocess_map = {
    "assist2009": preprocess_assist2009,
    "xes3g5m": preprocess_xes3g5m
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--raw_data_path", type=str,
                        default=r"F:\code\myProjects\pyktlib\example\data\raw\assist2009\skill_builder_data.csv")
    parser.add_argument("--save_dir", type=str,
                        default=r"F:\code\myProjects\pyktlib\example\data\preprocessed\assist2009")

    args = parser.parse_args()
    params = vars(args)
    data, Q_table = preprocess_map[params["dataset_name"]](params["raw_data_path"])
    write_kt_data(data, os.path.join(params["save_dir"], "data.txt"))
    np.save(os.path.join(params["save_dir"], "Q_table.npy"), Q_table)

    statics_qc = parse_Q_table(Q_table)
    statics_data = get_data_statics(data, statics_qc["num_question"])
    statics_qc.update(statics_data)
    write_json(statics_qc, os.path.join(params["save_dir"], "data_statics.json"))
