import torch
import torch.nn as nn
import sparseconvnet as scn

# two-dimensional SparseConvNet
class Model(nn.Module):
    def __init__(self, out_channels=256):
        nn.Module.__init__(self)
        self.sparseModel = scn.SparseVggNet(3, 3, [
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 48], ['C', 48], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 96], ['C', 96]]
        ).add(scn.Convolution(3, 96, 128, 3, 2, False)
        ).add(scn.BatchNormReLU(128)
        ).add(scn.SparseToDense(3, 128))
        #self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([4096, 4096]))
        self.inputLayer = scn.InputLayer(3,63,4)
        self.linear = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, 128)
        x = self.linear(x)
        return x
