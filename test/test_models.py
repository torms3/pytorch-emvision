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
        # print(y.size())

    def test_rsunet_gn(self):
        from emvision.models import rsunet_gn
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = rsunet_gn(width=[2,4,6,8], group=2).to(device)
        x = torch.randn(1,2,20,256,256).to(device)
        y = net(x)
        # print(y.size())

    def test_rsunet_act(self):
        from emvision.models import rsunet_act
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = rsunet_act(width=[2,4,6,8], act='PReLU').to(device)
        x = torch.randn(1,3,20,256,256).to(device)
        y = net(x)
        # print(y.size())

    def test_rsunet_act_gn(self):
        from emvision.models import rsunet_act_gn
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = rsunet_act_gn(width=[2,4,6,8], group=2, act='PReLU').to(device)
        x = torch.randn(1,2,20,256,256).to(device)
        y = net(x)
        # print(y.size())

    def test_rsunet_2d3d(self):
        from emvision.models import rsunet_2d3d
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = rsunet_2d3d(width=[3,4,5,6], depth2d=2).to(device)
        x = torch.randn(1,3,20,256,256).to(device)
        y = net(x)
        # print(y.size())

    def test_rsunet_2d3d_gn(self):
        from emvision.models import rsunet_2d3d_gn
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = rsunet_2d3d_gn(width=[2,4,6,8], group=2).to(device)
        x = torch.randn(1,2,20,256,256).to(device)
        y = net(x)
        # print(y.size())


if __name__ == '__main__':
    print('torch version =', torch.__version__)
    print('cuda  version =', torch.version.cuda)
    print('cudnn version =', torch.backends.cudnn.version())
    print('cuda available?', torch.cuda.is_available())
    unittest.main()
