import unittest
import torch
from my_mixkabrn_model.mixkabrn import MixKABRN

class TestMixKABRN(unittest.TestCase):
    def test_forward(self):
        model = MixKABRN(input_dim=128, output_dim=128)
        inputs = torch.randn(1, 10, 128)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 10, 128))

if __name__ == '__main__':
    unittest.main()

