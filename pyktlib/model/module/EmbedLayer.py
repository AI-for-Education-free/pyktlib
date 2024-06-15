import torch
import torch.nn as nn


class EmbedLayer(nn.Module):
    def __init__(self, embed_configs):
        super(EmbedLayer, self).__init__()

        for embed_config in embed_configs:
            embed_name = embed_config["embed_name"]
            num_item = embed_config["num_item"]
            dim_item = embed_config["dim_item"]
            learnable = embed_config["learnable"]
            if learnable:
                setattr(self, embed_name, nn.Embedding(num_item, dim_item))
            else:
                setattr(self, embed_name, EmbedLayer.init_constant_embed(embed_config))

    @staticmethod
    def init_constant_embed(embed_config):
        """
        初始化固定的embed layer，如下\n
        1、embed_correctness_method1：(2, dim)，0用全0表示，1用全1表示\n
        2、embed_correctness_method1：(2, dim)，0用左边一半元素为1，右边一半元素为0表示，1则相反

        :param embed_config:
        :return:
        """
        init_method = embed_config["init_method"]
        dim_item = embed_config["dim_item"]
        if init_method == "embed_correctness_method1":
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0] = torch.zeros(dim_item)
            embed.weight.data[1] = torch.ones(dim_item)
        elif init_method == "embed_correctness_method1":
            dim_half = dim_item // 2
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0, :dim_half] = 0
            embed.weight.data[0, dim_half:] = 1
            embed.weight.data[1, :dim_half] = 1
            embed.weight.data[1, :dim_half] = 0
        else:
            raise NotImplementedError()
        embed.weight.requires_grad = False
        return embed

    def get_emb(self, embed_name, item_index):
        """
        获取指定embed里的emb

        :param embed_name:
        :param item_index:
        :return:
        """
        embed = getattr(self, embed_name)
        return embed(item_index)

    def get_emb_all(self, embed_name):
        embed = getattr(self, embed_name)
        return embed.weight

    def get_emb_concatenated(self, cat_order, item_index2cat):
        """
        获取拼接后的emb，cat_order是拼接的顺序，item_index2cat是id序列（bs * seq_len）

        :param cat_order:
        :param item_index2cat:
        :return:
        """
        concatenated_emb = self.get_emb(cat_order[0], item_index2cat[0])
        for i, embed_name in enumerate(cat_order[1:]):
            concatenated_emb = torch.cat((concatenated_emb, self.get_emb(embed_name, item_index2cat[i + 1])), dim=-1)
        return concatenated_emb

    def get_emb_fused1(
            self,
            related_embed_name,
            base2related_transfer_table,
            base2related_mask_table,
            base_item_index,
            fusion_method="mean"
    ):
        """
        获取多个emb融合（如mean pool）后的emb

        例如一道习题关联多个知识点，则首先根据习题id找到对应的多个知识点id，然后取出对应的知识点emb并fuse

        在上面那个例子中，将习题记为base，知识点记为related

        base和related是1对多的关系

        :param related_embed_name:
        :param base2related_transfer_table:
        :param base2related_mask_table:
        :param base_item_index:
        :param fusion_method:
        :return:
        """
        embed_related = getattr(self, related_embed_name)
        related_emb = embed_related(base2related_transfer_table[base_item_index])
        mask = base2related_mask_table[base_item_index]
        if fusion_method == "mean":
            related_emb_fusion = (related_emb * mask.unsqueeze(-1)).sum(-2)
            related_emb_fusion = related_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related_emb_fusion

    def get_emb_fused2(
            self,
            related2_embed_name,
            base2related1_transfer_table,
            base2related1_mask_table,
            related_1_to_2_transfer_table,
            base_item_index,
            fusion_method="mean"
    ):
        """
        获取多个emb融合（如mean pool）后的emb

        例如一道习题关联多个知识点，每个知识点又关联一个知识点难度，则首先根据习题id找到对应的多个知识点id，然后取出对应的知识点难度id，
        并取出知识点难度emb，然后fuse

        在上面那个例子中，将习题记为base，知识点记为related1，知识点难度记为related2

        base和related是1对多的关系，related1和related2是1对1的关系

        :param related2_embed_name:
        :param base2related1_transfer_table:
        :param base2related1_mask_table:
        :param related_1_to_2_transfer_table: 以上面例子为基础，假设有K个知识点，k个知识点难度，则该table为K * 1 tensor，
        里面元素是知识点id和知识点难度id的对应关系，即other_table[k]是第k个concept对应的知识点难度id
        :param base_item_index:
        :param fusion_method:
        :return:
        """
        embed_related2 = getattr(self, related2_embed_name)
        related1_item_index = base2related1_transfer_table[base_item_index]
        related2_emb = embed_related2(related_1_to_2_transfer_table[related1_item_index])
        mask = base2related1_mask_table[base_item_index]
        if fusion_method == "mean":
            related2_emb_fusion = (related2_emb * mask.unsqueeze(-1)).sum(-2)
            related2_emb_fusion = related2_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related2_emb_fusion
