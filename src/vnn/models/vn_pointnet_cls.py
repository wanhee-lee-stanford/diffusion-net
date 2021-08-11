import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.vn_pointnet import PointNetEncoder

class get_model(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=True):
        super(get_model, self).__init__()
        self.args = args
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(x.shape, 'shape before pointnet')
        x, trans, trans_feat = self.feat(x)
        print(x.shape, 'self.feat(x) shape')
        print(trans.shape, 'self.feat(trans) shape')
        try:
            print(trans_feat.shape, 'self.feat(trans_feat) shape')
        except:
            pass
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        print(x.shape, 'vn_point_cls model output x.shape')
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        return loss
