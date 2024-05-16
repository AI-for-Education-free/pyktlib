class ConfigureParamNotExist(Exception):
    def __init__(self, msg="param not exist"):
        super(ConfigureParamNotExist, self).__init__()
        self.message = msg
