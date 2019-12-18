import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from feat.dataloader import provider
import scipy.ndimage
import math

warnings.filterwarnings('ignore')

DATA_PATH = 'data/modelnet40_normal_resampled/'
elastic_deformation=False
full_scale=255 #Input field size
scale=20  #Voxel size = 1/scale

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, split, args, root=DATA_PATH,  npoint=1024, uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'val' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.label = [self.classes[self.datapath[index][0]] for index in range(len(self.datapath))]


        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.label[index]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            '''
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]
            '''
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

            a = point_set[:,:3]
            m=np.eye(3)+np.random.randn(3,3)*0.1
            m[0][0]*=np.random.randint(0,2)*2-1
            m*=scale
            #theta=np.random.rand()*2*math.pi
            #m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
            a=np.matmul(a,m)
            if elastic_deformation:
                a=self.elastic(a,6*scale//50,40*scale/50)
                a=self.elastic(a,20*scale//50,160*scale/50)
            m=a.min(0)
            M=a.max(0)
            offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
            a+=offset
            point_set[:,:3] = a
            idxs=(a.min(1)>=0)*(a.max(1)<full_scale)

            point_set = point_set[idxs]

            '''
            points = provider.random_point_dropout(np.expand_dims(point_set, 0))
            points[:,:, 0:3] = provider.random_scale_point_cloud(np.expand_dims(point_set, 0)[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(np.expand_dims(point_set, 0)[:,:, 0:3])   
            points = points[0]             
            '''

        return point_set, cls


    def __getitem__(self, index):
        return self._get_item(index)

    def elastic(self,x,gran,mag):
        bb=np.abs(x).max(0).astype(np.int32)//gran+3
        noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
        noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
        noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
        noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
        noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
        noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
        noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
        ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
        interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x+g(x)*mag



if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
