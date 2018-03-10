import torch
import emvision
import unittest


class Tester(unittest.TestCase):

    def test_rsunet(self):
        from emvision.models import RSUNet
        model = RSUNet()


if __name__ == '__main__':
    unittest.main()
