import unittest
import torch
from pyktlib.model.module.EmbedLayer import EmbedLayer


embed_configs = [
    {
        "embed_name": "concept",
        "num_item": 123,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "question",
        "num_item": 17751,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "correctness",
        "num_item": 2,
        "dim_item": 64,
        "learnable": False,
        "init_method": "embed_correctness_method1"
    },
    {
        "embed_name": "concept_diff",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "question_diff",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "use_time",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "interval_time",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "num_hint",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    },
    {
        "embed_name": "num_attempt",
        "num_item": 100,
        "dim_item": 64,
        "learnable": True
    }
]


class TestEmbedLayer(unittest.TestCase):
    def test_get_emb(self):
        embed_layer = EmbedLayer(embed_configs)
        concept_seq = torch.tensor([[1, 2, 3], [4, 5, 6]]).long()
        concept_emb = embed_layer.get_emb("concept", concept_seq)
        bs, seq_len, dim = concept_emb.shape
        self.assertTrue((bs == 2) and (seq_len == 3) and (dim == 64))

    def test_get_concatenated_emb(self):
        embed_layer = EmbedLayer(embed_configs)
        concept_seq = torch.tensor([[1, 2, 3], [4, 5, 6]]).long()
        question_seq = torch.tensor([[1, 2, 3], [4, 5, 6]]).long()
        correctness_seq = torch.tensor([[1, 0, 1], [0, 1, 1]]).long()
        interaction_emb = embed_layer.get_emb_concatenated(
            ["concept", "question", "correctness"], [concept_seq, question_seq, correctness_seq]
        )
        bs, seq_len, dim = interaction_emb.shape
        self.assertTrue((bs == 2) and (seq_len == 3) and (dim == (64 * 3)))


if __name__ == "__main__":
    unittest.main()
