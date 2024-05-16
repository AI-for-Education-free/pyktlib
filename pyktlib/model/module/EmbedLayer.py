import torch.nn as nn


class EmbedLayer(nn.Module):
    def __init__(self):
        super(EmbedLayer, self).__init__()
        print("init")
        