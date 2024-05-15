import unittest
from pyktlib.model.module.EmbedLayer import EmbedLayer


class TestEmbedLayer(unittest.TestCase):
    def test_init(self):
        embed_layer = EmbedLayer()
        self.assertEqual(type(embed_layer).__name__, "EmbedLayer")


if __name__ == "__main__":
    unittest.main()
