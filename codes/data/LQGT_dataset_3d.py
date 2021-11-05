import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import logging

from . import util


class LQGTDataset3D(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT vti file pairs.
    If only GT image is provided, generate LQ vti on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    logger = logging.getLogger('base')

    def __init__(self, opt):
        super(LQGTDataset3D, self).__init__()
        self.opt = opt
        self.paths_LQ, self.paths_GT = None, None

        if opt['type'] == 'vtk':
            self.paths_GT = util.get_vtk_paths(opt['dataroot_GT'])
            self.paths_LQ = util.get_vtk_paths(opt['dataroot_LQ'])
        elif opt['type'] == 'tecplot':
            self.paths_GT = util.get_tecplot_paths(opt['dataroot_GT'])
            self.paths_LQ = util.get_tecplot_paths(opt['dataroot_LQ'])
        else:
            ex = Exception("Type '%s' is not supported" % opt['type'])
            raise ex

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        attr_id = self.opt['attr_id']

        # get GT image
        GT_path = self.paths_GT[index]
        vti_GT_generator = util.getTensorGenerator(GT_path)
        vti_GT_generator.set_type(self.opt['type'])
        vti_GT, component_GT = vti_GT_generator.get_numpy_array(attr_id)
        if self.opt['phase'] != 'train':
            vti_GT = util.modcrop_3d(vti_GT, scale)

        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            vti_LQ_generator = util.getTensorGenerator(LQ_path)
            vti_LQ_generator.set_type(self.opt['type'])
            vti_LQ, component_LQ = vti_LQ_generator.get_numpy_array(attr_id)
        else:
            if self.opt['phase'] == 'train':
                # random_scale = random.choice(self.random_scale_list)
                # Z_s, Y_s, X_s = vti_GT.shape

                # def _mod(n, random_scale, scale, thres):
                #     rlt = int(n * random_scale)
                #     rlt = (rlt // scale) * scale
                #     return thres if rlt < thres else rlt

                # Z_s = _mod(Z_s, random_scale, scale, GT_size)
                # Y_s = _mod(Y_s, random_scale, scale, GT_size)
                # X_s = _mod(X_s, random_scale, scale, GT_size)
                vti_GT = util.resize_3d(arr=np.copy(vti_GT), newsize=GT_size)

            # using matlab imresize3
            vti_LQ = util.imresize3_np(vti_GT, 1 / scale, True)
            if vti_LQ.ndim != 3:
                ex = Exception("Error: dims not right")
                raise ex

        if self.opt['phase'] == 'train':
            Z, Y, X = vti_GT.shape
            if Z < GT_size or Y < GT_size or X < GT_size:
                vti_GT = util.resize_3d(np.copy(vti_GT), newsize=GT_size)
                # using matlab imresize3
                vti_LQ = util.imresize3_np(vti_GT, 1 / scale, True)
                if vti_LQ.ndim != 2:
                    ex = Exception("Error: dims not right")
                    raise ex

            Z, Y, X = vti_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_Z = random.randint(0, max(0, Z - LQ_size))
            rnd_Y = random.randint(0, max(0, Y - LQ_size))
            rnd_X = random.randint(0, max(0, X - LQ_size))
            vti_LQ = vti_LQ[rnd_Z: rnd_Z + LQ_size, rnd_Y: rnd_Y + LQ_size, rnd_X: rnd_X + LQ_size]
            rnd_Z_GT, rnd_Y_GT, rnd_X_GT = int(rnd_Z * scale), int(rnd_Y * scale), int(rnd_X * scale)
            vti_GT = vti_GT[rnd_Z_GT: rnd_Z_GT + GT_size, rnd_Y_GT: rnd_Y_GT + GT_size, rnd_X_GT: rnd_X_GT + GT_size]

        # ZYX to XYZ
        vti_GT = torch.from_numpy(np.ascontiguousarray(vti_GT)).float().unsqueeze(0)
        vti_LQ = torch.from_numpy(np.ascontiguousarray(vti_LQ)).float().unsqueeze(0)


        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': vti_LQ, 'GT': vti_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
