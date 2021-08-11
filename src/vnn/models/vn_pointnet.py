import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross


class STNkd(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        print(x.shape, 'x.shape on vn_pointnet PointNetEncoder forward first unsqueeze')
        feat = get_graph_feature_cross(x, k=self.n_knn)
        print(feat.shape, 'feat.shape on vn_pointnet PointNetEncoder forward after get_graph_feature_cross')
        x = self.conv_pos(feat)
        print(x.shape, 'x.shape on vn_pointnet PointNetEncoder forward after conv_pos(feat)')
        x = self.pool(x)
        print(x.shape, 'x.shape on vn_pointnet PointNetEncoder forward after pool(x)')
        
        x = self.conv1(x)
        print(x.shape, 'x.shape on vn_pointnet PointNetEncoder forward after self.conv1')
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)

        print(x.shape, 'x.shape on vn_pointnet after feature transform')
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        print(x.shape, 'x.shape on vn_pointnet before x_mean torch cat')
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        print(x.shape, 'x.shape on vn_pointnet after std_feature')

        x = x.view(B, -1, N)
        print(x.shape,'x.shape on vn_pointnet after x.view(B, -1, N)')
        
        x = torch.max(x, -1, keepdim=False)[0]
        print(x.shape, 'x.shape on vn_pointnet after torch.max(x, -1, keepdim=False)[0]')

        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
