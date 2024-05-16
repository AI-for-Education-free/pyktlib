from typing import Union, List, Tuple, Dict
from copy import deepcopy

from ..util.exception.ConfigureParamNotExist import ConfigureParamNotExist


class BasicConfigure:
    def __init__(self):
        self.params = {}
        self.error_messages = {
            "key_chain_len": "if depth smaller than 1, use str as key_chain",
            "set_not_last_depth": "you can only set the last depth key"
        }

    def add_param(self, key_chain: Union[str, List[str], Tuple[str]], value: Union[str, int, float, dict]):
        if type(key_chain) is str:
            self.params[key_chain] = value if (type(value) is not dict) else deepcopy(dict)
        else:
            assert len(key_chain) > 1, self.error_messages["key_chain_len"]
            params = self.params
            for k in key_chain[:-1]:
                params.setdefault(k, {})
                params = params[k]
            params[key_chain[-1]] = value if (type(value) is not dict) else deepcopy(value)

    def get_param(self, key_chain: Union[str, List[str], Tuple[str]]):
        if type(key_chain) is str:
            if key_chain not in self.params.keys():
                raise ConfigureParamNotExist(f"params[{key_chain}] not exist")
            else:
                return self.params[key_chain]
        else:
            assert len(key_chain) > 1, self.error_messages["key_chain_len"]
            params = self.params
            params_str = "params"
            for k in key_chain[:-1]:
                params_str = f"{params_str}[{k}]"
                if k not in params.keys():
                    raise ConfigureParamNotExist(f"{params_str} not exist")
                params = params[k]
            last_key = key_chain[-1]
            if last_key not in params.keys():
                raise ConfigureParamNotExist(f"{params_str}[{last_key}] not exist")
            else:
                value = params[last_key]
                return value if (type(value) is not dict) else deepcopy(value)

    def set_param(self, key_chain: Union[str, List[str], Tuple[str]], value: Union[str, int, float]):
        if type(key_chain) is str:
            self.params[key_chain] = value
        else:
            assert len(key_chain) > 1, self.error_messages["key_chain_len"]
            params = self.params
            params_str = "params"
            for k in key_chain[:-1]:
                params_str = f"{params_str}[{k}]"
                if k not in params.keys():
                    raise ConfigureParamNotExist(f"{params_str} not exist")
                params = params[k]
            assert type(params[key_chain[-1]]) in [str, int, float], self.error_messages["set_not_last_depth"]
            params[key_chain[-1]] = value

    @staticmethod
    def dict2list(dict_param: Dict[int, Union[str, int, float]]):
        kv_tuple = list(map(lambda x: (x[0], x[1]), dict_param.items()))
        result = list(map(lambda x: x[1], sorted(kv_tuple, key=lambda x: x[0])))
        return result

    @staticmethod
    def list2dict(list_param: Union[List[Union[str, int, float]], Tuple[Union[str, int, float]]]):
        result = {k: v for k, v in enumerate(list_param)}
        return result
