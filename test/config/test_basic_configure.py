import unittest
from pyktlib.config.BasicConfigure import BasicConfigure


class TestBasicConfigure(unittest.TestCase):
    def test_add_param(self):
        configure = BasicConfigure()

        key_chain1 = "device"
        key_chain2 = ["train_strategy", "type"]
        key_chain3 = ["models", "kt_model"]
        key_chain4 = ["train_strategy", "multi_metrics"]

        value1 = "cuda"
        value2 = "valid_test"
        value3 = {
            "encoder": {
                "num_concept": 123,
                "num_question": 17751
            }
        }
        value4 = BasicConfigure.list2dict(['AUC'])

        configure.add_param(key_chain1, value1)
        configure.add_param(key_chain2, value2)
        configure.add_param(key_chain3, value3)
        configure.add_param(key_chain4, value4)

        value1_got = configure.get_param(key_chain1)
        value2_got = configure.get_param(key_chain2)
        value3_got = configure.get_param(key_chain3)
        value4_got = configure.get_param(key_chain4)

        self.assertTrue(value1 == value1_got)
        self.assertTrue(value2 == value2_got)
        self.assertTrue(value3 == value3_got)
        self.assertTrue(value4 == value4_got)


if __name__ == "__main__":
    unittest.main()
