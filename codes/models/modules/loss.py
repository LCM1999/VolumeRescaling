import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from codes.models.modules.Subnet_constructor import UpsampleBlock
from codes.models.modules.SSIM_3d import ssim3D
from codes.models.modules.Sobel_3d import sobel_edge_3d
from codes.models.modules.MarchingCubes import MarchingCubes
from codes.models.modules.DistanceField import DistanceField
import codes.models.modules.funcs as funcs

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, wait


class BorderLoss(nn.Module):
    def __init__(self):
        super(BorderLoss, self).__init__()
        self.upsampler = UpsampleBlock()

    def forward(self, origin, x):
        upper = (origin.max() - x.max()) ** 2
        lower = (origin.min() - x.min()) ** 2
        print("o_min: {}, o_max: {}, x_min: {}, x_max: {}, upper: {}, lower: {}".format(
            origin.min(), origin.max(), x.min(), x.max(), upper, lower))
        return upper + lower
        # if not isLR:
        #     return abs((origin - x).max())
        # else:
        #     upsampled = self.upsampler(x)
        #     return abs((origin - upsampled).max())


class FidelityLoss(nn.Module):
    def __init__(self):
        super(FidelityLoss, self).__init__()
        self.upsampler = UpsampleBlock()
        self.normloss = nn.L1Loss()
        # self.ssim = SSIM3D()

    def forward(self, origin, x, isLR=True, addNormLoss=True, addSSIM=False):
        ssimloss = 0
        normloss = 0
        if not isLR:
            if addSSIM:
                ssimloss = 2 - (ssim3D(origin, x) + 1)
            if addNormLoss:
                normloss = self.normloss(origin, x)
            return ssimloss, normloss
        else:
            upsampled = self.upsampler(x)
            if addSSIM:
                ssimloss = 2 - (ssim3D(origin, upsampled) + 1)
            if addNormLoss:
                normloss = self.normloss(origin, upsampled)
            return ssimloss, normloss


class GradientLoss(nn.Module):
    def __init__(self, scale):
        super(GradientLoss, self).__init__()
        self.scale = scale
        #
        # sobel1 = torch.tensor([1, 0, -1], dtype=torch.float)
        # sobel2 = torch.tensor([1, 2, 1], dtype=torch.float)
        #
        # sobel_weight1 = [sobel1, sobel1, sobel1]
        # sobel_weight2 = [sobel2, sobel2, sobel2]
        #
        # for xyz in range(3):
        #     newShape = [1, 1, 1, 1, 1]
        #     newShape[xyz + 2] = 3
        #     sobel_weight1[xyz] = torch.reshape(sobel1, newShape)
        #     sobel_weight2[xyz] = torch.reshape(sobel2, newShape)
        #
        # conv_x = []
        # conv_y = []
        # conv_z = []
        #
        # tmpconv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
        #
        # tmpconv.weight = nn.Parameter(sobel_weight1[0])
        # conv_x.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[1])
        # conv_x.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[2])
        # conv_x.append(deepcopy(tmpconv))
        #
        # tmpconv.weight = nn.Parameter(sobel_weight1[1])
        # conv_y.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[0])
        # conv_y.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[2])
        # conv_y.append(deepcopy(tmpconv))
        #
        # tmpconv.weight = nn.Parameter(sobel_weight1[2])
        # conv_z.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[0])
        # conv_z.append(deepcopy(tmpconv))
        # tmpconv.weight = nn.Parameter(sobel_weight2[1])
        # conv_z.append(deepcopy(tmpconv))
        #
        # self.conv_x = nn.Sequential(*conv_x)
        # self.conv_y = nn.Sequential(*conv_y)
        # self.conv_z = nn.Sequential(*conv_z)

    def forward(self, o, x, isLR=True):
        gradient_x = sobel_edge_3d(x.cpu())
        # gradient_x = torch.cat([self.conv_x(x), self.conv_y(x), self.conv_z(x)], 1).cpu()
        l0_x = int(torch.norm(input=gradient_x, p=0).detach().numpy())
        gradient_o = sobel_edge_3d(o.cpu())
        # gradient_o = torch.cat([self.conv_x(o), self.conv_y(o), self.conv_z(o)], 1).cpu()
        l0_o = int(torch.norm(input=gradient_o, p=0).detach().numpy())
        size_o = gradient_o.numel()
        print("l0: x: {}, o: {} \n size_o: {}".format(l0_x, l0_o, size_o))
        if isLR:
            return abs((self.scale ** 3 * l0_x - l0_o) / size_o)
        else:
            return abs((l0_x - l0_o) / size_o)


def _calculate_distanceFields_h(field: DistanceField):
    field.CalculateDistanceField()
    return field


def float_distance_square(v1, v2):
    x_2 = (v2[0] - v1[0]) ** 2
    y_2 = (v2[1] - v1[1]) ** 2
    z_2 = (v2[2] - v1[2]) ** 2
    return x_2 + y_2 + z_2


class IsosurfaceSimilarityLoss:
    def __init__(self, data_type, isovalueNum=6, scale=1, DfDownscale=8, numSamples=1500, histSize=128, approx=True):
        self.IsovalueNum = isovalueNum
        self.scale = scale
        self.DataType = data_type
        self.DfDownscale = DfDownscale
        self.NUM_SAMPLES = numSamples
        self.HIST_SIZE = histSize
        self.Approx = approx
        self.shape = None
        self.GT = None
        self.HR = None
        self.loss = None
        if self.Approx:
            self.loss = nn.L1Loss()

    def setGTHR(self, GT, HR, isTensor=True):
        if isTensor:
            self.GT = torch.squeeze(GT.cpu()).detach().numpy()
            self.HR = torch.squeeze(HR.cpu()).detach().numpy()
        else:
            self.GT = GT
            self.HR = HR
        if self.DataType == 'cell':
            self.GT, _ = funcs.ConvertCellDataToPoints(tuples=self.GT.size, cellData=self.GT)
            self.HR, _ = funcs.ConvertCellDataToPoints(tuples=self.HR.size, cellData=self.HR)
        if self.GT.shape == self.HR.shape:
            self.shape = self.GT.shape
        else:
            raise Exception("GT shape {} != HR shape {}".format(self.GT.shape, self.HR.shape))

    def _findBucket_d(self, val, minVal, step, id1, id2):
        if step == 0:
            return 0
        bucket = int((val - minVal) / step)
        return (self.HIST_SIZE - 1) if bucket >= self.HIST_SIZE else bucket

    def _calculate_isovalues(self):
        GT_min = np.min(a=self.GT)
        GT_max = np.max(a=self.GT)
        print("GT max: {}, min: {}".format(GT_max, GT_min))
        HR_min = np.min(a=self.HR)
        HR_max = np.max(a=self.HR)
        print("HR max: {}, min: {}".format(HR_max, HR_min))
        GT_isovalues = np.linspace(start=GT_min,
                                   stop=GT_max,
                                   num=(self.IsovalueNum + 1),
                                   endpoint=False).tolist()[1:]
        isovalues = []
        for value in GT_isovalues:
            if HR_min < value < HR_max:
                isovalues.append(value)
        if len(isovalues) == 0:
            print("value: {}, HR_min: {}, HR_max: {}".format(GT_isovalues[0], HR_min, HR_max))
            return None
        # HR_isovalues = np.linspace(start=HR_min,
        #                            stop=HR_max,
        #                            num=(self.IsovalueNum + 1),
        #                            endpoint=False).tolist()[1:]
        return GT_isovalues

    def _calculate_surfaces(self):
        print("Calculating Isosurfaces")
        sampleSurfaces_GT = []
        sampleSurfaces_HR = []
        GT_isovalues = self._calculate_isovalues()
        if GT_isovalues is None:
            return None, None
        print(GT_isovalues)
        for GT_isovalue in GT_isovalues:
            print(len(sampleSurfaces_HR))
            approxPoints_GT = []
            approxPoints_HR = []
            if self.Approx:
                approxPoints_GT = \
                    funcs.ApproximateIsosurfaces_CPU(h_data=funcs.getFlatVolume(self.GT),
                                                     curIsovalue=GT_isovalue,
                                                     dims=self.GT.shape,
                                                     scale=self.scale)
                approxPoints_HR = \
                    funcs.ApproximateIsosurfaces_CPU(h_data=funcs.getFlatVolume(self.HR),
                                                     curIsovalue=GT_isovalue,
                                                     dims=self.HR.shape,
                                                     scale=self.scale)
            else:
                marchingCubes_GT = MarchingCubes(volume=self.GT, isovalue=GT_isovalue)
                marchingCubes_HR = MarchingCubes(volume=self.HR, isovalue=GT_isovalue)
                vertices_GT = marchingCubes_GT.marching_cube()
                vertices_HR = marchingCubes_HR.marching_cube()
                for vertex_GT, vertex_HR in vertices_GT, vertices_HR:
                    approxPoints_GT.append(tuple(deepcopy(vertex_GT.v)))
                    approxPoints_HR.append(tuple(deepcopy(vertex_HR.v)))
            sampleSurfaces_GT.append(approxPoints_GT)
            sampleSurfaces_HR.append(approxPoints_HR)
        print("Calculating Isosurfaces Finished")
        return sampleSurfaces_GT, sampleSurfaces_HR

    def _calculate_Approx_field(self, surface):
        dscale = self.DfDownscale
        single_dims = (self.shape[0] // dscale,
                       self.shape[1] // dscale,
                       self.shape[2] // dscale)
        field = np.zeros(single_dims)
        for i in range(single_dims[0]):
            for j in range(single_dims[1]):
                for k in range(single_dims[2]):
                    cur = sys.float_info.max
                    for point in surface:
                        distance_square = float_distance_square((i * dscale + dscale / 2,
                                                                 j * dscale + dscale / 2,
                                                                 k * dscale + dscale / 2),
                                                                point)
                        if distance_square < cur:
                            field[i][j][k] = distance_square
                            cur = distance_square
        return field

    def _calculate_distance_fields_APPROX(self, surfaces_GT, surfaces_HR):
        print("Calculating Distance Fields APPROX")
        fields_GT = []
        fields_HR = []
        for sampleSurface_GT, sampleSurface_HR in zip(surfaces_GT, surfaces_HR):
            fields_GT.append(self._calculate_Approx_field(sampleSurface_GT))
            fields_HR.append(self._calculate_Approx_field(sampleSurface_HR))
        print("Calculating Distance Fields APPROX Finished")
        print("Calculating L1Loss")
        fields_GT_tensor = torch.tensor(fields_GT)
        fields_HR_tensor = torch.tensor(fields_HR)
        loss = self.loss.forward(fields_GT_tensor, fields_HR_tensor)
        print("Calculating L1Loss Finished")
        return loss

    def _calculate_distanceFields(self, surfaces_GT, surfaces_HR):
        print("Calculating Distance Fields")
        fields_GT = []
        fields_HR = []
        for sampleSurface_GT, sampleSurface_HR in zip(surfaces_GT, surfaces_HR):
            dims_GT = self.GT.shape
            dims_HR = self.HR.shape
            fields_GT.append(DistanceField(points=sampleSurface_GT,
                                           dims=(dims_GT[0] // self.DfDownscale,
                                                 dims_GT[1] // self.DfDownscale,
                                                 dims_GT[2] // self.DfDownscale),
                                           dfDownscale=self.DfDownscale,
                                           dimsOrigin=dims_GT))
            fields_HR.append(DistanceField(points=sampleSurface_HR,
                                           dims=(dims_HR[0] // self.DfDownscale,
                                                 dims_HR[1] // self.DfDownscale,
                                                 dims_HR[2] // self.DfDownscale),
                                           dfDownscale=self.DfDownscale,
                                           dimsOrigin=dims_HR))

            fields_GT[-1].CalculateDistanceField(numSamples=self.NUM_SAMPLES)
            fields_HR[-1].CalculateDistanceField(numSamples=self.NUM_SAMPLES)
        # refield_GT = []
        # refield_HR = []
        # with ProcessPoolExecutor(max_workers=8) as pool:
        #     for field in pool.map(_calculate_distanceFields_h, fields_GT):
        #         refield_GT.append(field)
        #     for field in pool.map(_calculate_distanceFields_h, fields_HR):
        #         refield_HR.append(field)
        # del fields_GT
        # del fields_HR
        print("Calculating Distance Fields Finished")
        return fields_GT, fields_HR

    def _calculate_mutualInfo(self, hist, fieldSize, colSums, rowSums):
        hX = 0
        hY = 0
        hXY = 0
        for i in range(self.HIST_SIZE):
            for j in range(self.HIST_SIZE):
                if hist[i * self.HIST_SIZE + j] > 0:
                    pxy = hist[i * self.HIST_SIZE + j]
                    hXY -= pxy * np.log(pxy).astype('float32')
            if colSums[i] > 0:
                px = colSums[i]
                hX -= px * np.log(px).astype('float32')
            if rowSums[i] > 0:
                py = rowSums[i]
                hY -= py * np.log(py).astype('float32')
        hXY = hXY / fieldSize + np.log(fieldSize).astype('float32')
        hX = hX / fieldSize + np.log(fieldSize).astype('float32')
        hY = hY / fieldSize + np.log(fieldSize).astype('float32')
        iXY = hX + hY - hXY
        val = 2 * iXY / (hX + hY)
        return 0.0 if val != val else val

    def _calculate_similarityMap(self, dfields_GT, dfields_HR, fieldSize):
        print("Calculating Similarity")
        histSize2 = self.HIST_SIZE ** 2
        numFields2 = self.IsovalueNum ** 2
        jointHist = np.zeros((numFields2, histSize2), dtype='int')
        colRowSize = (numFields2, self.HIST_SIZE)
        colSums = np.zeros(colRowSize, dtype='int')
        rowSums = np.zeros(colRowSize, dtype='int')
        simMap = np.zeros(numFields2, dtype='float32')
        for i in range(self.IsovalueNum):
            for j in range(self.IsovalueNum):
                field1 = dfields_GT[i].getDistanceArray()
                field2 = dfields_HR[j].getDistanceArray()
                minVal = min(dfields_GT[i].getMinVal(), dfields_HR[j].getMinVal())
                maxVal = max(dfields_GT[i].getMaxVal(), dfields_HR[j].getMaxVal())
                step = (maxVal - minVal) / self.HIST_SIZE
                # idx1 = i * self.IsovalueNum
                # idx2 = j
                idx = i * self.IsovalueNum + j
                # histIdx = i * self.IsovalueNum + j
                # colRowIdx = i * self.IsovalueNum + j
                # print("histIdx: {}, colrowIdx: {}".format(histIdx, colRowIdx))

                for k in range(fieldSize):
                    row = self._findBucket_d(field1[k], minVal, step, i, j)
                    column = self._findBucket_d(field2[k], minVal, step, i, j)
                    # print("row: {}, col: {}".format(row, column))
                    jointHist[idx][row * self.HIST_SIZE + column] += 1
                    colSums[idx][column] += 1
                    rowSums[idx][row] += 1
                # print("JointHist: {}, \n colSums: {}, \n rowSums: {}".format(
                #     jointHist, colSums, rowSums))

                simMap[i * self.IsovalueNum + j] = \
                    self._calculate_mutualInfo(hist=jointHist[idx],
                                               fieldSize=self.GT.size,
                                               colSums=colSums[idx],
                                               rowSums=rowSums[idx])
        print("Calculating Similarity Finished")
        return simMap

    def _calculate_similarity(self):
        sampleSurfaces_GT, sampleSurfaces_HR = self._calculate_surfaces()
        if sampleSurfaces_GT is None or sampleSurfaces_HR is None:
            return None
        if self.Approx:
            loss = self._calculate_distance_fields_APPROX(sampleSurfaces_GT, sampleSurfaces_HR)
            return loss
        fields_GT, fields_HR = self._calculate_distanceFields(sampleSurfaces_GT, sampleSurfaces_HR)
        dims = self.GT.shape
        fieldSize = (dims[0] // self.DfDownscale) * \
                    (dims[1] // self.DfDownscale) * \
                    (dims[2] // self.DfDownscale)
        simMap = self._calculate_similarityMap(fields_GT, fields_HR, fieldSize)
        print(simMap)
        return np.min(simMap)

    def forward(self):
        res = self._calculate_similarity()
        if res is None:
            return 1
        elif self.Approx:
            return res
        else:
            return (1 - res) * 1000
        # return 1 - self._calculate_similarity()


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3, 4)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3, 4)))
        else:
            print("reconstruction loss type error!")
            return 0


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


if __name__ == '__main__':
    import codes.data.tensor_generator as Reader
    from funcs import ConvertCellDataToPoints
    from copy import deepcopy

    reader = Reader.TensorGenerator()
    reader.set_path("D:\\testHalfTL\\test_0.vti")
    reader.set_type('point')
    # reader.set_path("E:\\JHTDB\\isotropic1024coarse\\p\\isotropic1024coarse_p_128_0.vti")
    reader.update()
    cellGT, _ = reader.get_array_by_id(index=0)
    print(cellGT.size)
    reader.set_path("D:\\testHalfTL\\isotropic1024coarse_p_128_0_100.vti")
    reader.update()
    cellHR, _ = reader.get_array_by_id(index=0)
    # cellHR = deepcopy(cellGT)
    print(cellHR.size)

    # pointGT, _ = ConvertCellDataToPoints(tuples=cellGT.size, cellData=cellGT)
    # pointHR, _ = ConvertCellDataToPoints(tuples=cellHR.size, cellData=cellHR)

    IsoLossTest = IsosurfaceSimilarityLoss(data_type='point', scale=8)
    IsoLossTest.setGTHR(cellGT, cellHR, isTensor=False)
    print(IsoLossTest.forward())
