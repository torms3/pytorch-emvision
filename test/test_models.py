import torch
import emvision
import unittest


class Tester(unittest.TestCase):

    def test_rsunet(self):
        from emvision.models import RSUNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = RSUNet(width=[3,4,5,6]).to(device)
        x = torch.randn(1,3,20,256,256).to(device)
        y = net(x)
        print(y.size())


if __name__ == '__main__':
    print('torch version=', torch.__version__)
    print('cuda version=', torch.version.cuda)
    print('cudnn version=', torch.backends.cudnn.version())
    print('cuda available? ', torch.cuda.is_available())
    unittest.main()
