import torch
import numpy as np

from copy import deepcopy

from MarchingCubes import MarchingCubes
import funcs as F


class Isosurface:
    def __init__(self, dims):
        self.dims = dims
        self.volume = None
        self.minV = None
        self.maxV = None
        self.volumeSize = None
        self.points = []

    def SetVolume(self, volume: torch.Tensor):
        self.volume = volume.detach().numpy()

    def Update(self):
        if self.volume is None:
            raise Exception("Volume have not set")

        self.minV = np.min(self.volume)
        self.maxV = np.max(self.volume)
        self.volumeSize = self.dims[0] * self.dims[1] * self.dims[2]

    def CalculateSurfacePoints(self, curIsovalue, approx=True):
        marchingCubes = MarchingCubes(volume=self, isovalue=curIsovalue)
        approxPoints = None

        if not approx:
            vertices = marchingCubes.marching_cube()
            for vertex in vertices:
                self.points.append(tuple(deepcopy(vertex.v)))

            length = len(self.points)
        else:
            scale = 1
            approxPoints = F.ApproximateIsosurfaces_CPU(h_data=F.getFlatVolume(self.volume),
                                                        curIsovalue=curIsovalue,
                                                        dims=self.dims,
                                                        scale=scale)
            length = len(approxPoints)
        return length, approxPoints
