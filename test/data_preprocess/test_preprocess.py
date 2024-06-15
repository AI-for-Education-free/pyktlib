import unittest
from pyktlib.data_preprocess.xes3g5m import preprocess_xes3g5m
from pyktlib.data_preprocess.assist2009 import preprocess_assist2009
from pyktlib.util.data.parse import check_Q_table


class TestPreprocess(unittest.TestCase):
    def test_preprocess_xes3g5m(self):
        data_uniformed, Q_table = preprocess_xes3g5m(r"F:\code\myProjects\pyktlib\example\data\raw\xes3g5m")
        self.assertTrue(check_Q_table(Q_table))

    def test_preprocess_assist2009(self):
        data_uniformed, Q_table = preprocess_assist2009(r"F:\code\myProjects\pyktlib\example\data\raw\assist2009\skill_builder_data.csv")
        self.assertTrue(check_Q_table(Q_table))


if __name__ == "__main__":
    unittest.main()
