import numpy as np

import sys


def float_distance(v1, v2):
    x_2 = (v2[0] - v1[0]) ** 2
    y_2 = (v2[1] - v1[1]) ** 2
    z_2 = (v2[2] - v1[2]) ** 2
    return np.sqrt(x_2 + y_2 + z_2)


class DistanceField:
    def __init__(self, dims=None, dfDownscale=None, points=None, dimsOrigin=None):
        self.dims = dims
        self.dimsOrigin = dimsOrigin
        self.points = points
        self.dfDownscale = dfDownscale
        self.distances = None
        self.id = None

    def SetId(self, id):
        self.id = id

    def CalculateDistanceField(self, numSamples=None, CPU=True, approx=True):
        dscale = self.dfDownscale
        print(self.dims)
        self.distances = np.zeros(self.dims)
        if CPU:
            for z in range(self.dims[0]):
                for y in range(self.dims[1]):
                    for x in range(self.dims[2]):
                        cur = sys.float_info.max
                        for point in self.points:
                            distance = float_distance(
                                (z * dscale + dscale / 2, y * dscale + dscale / 2, x * dscale + dscale / 2),
                                point)
                            if distance < cur:
                                self.distances[z][y][x] = distance
                                cur = distance

        else:
            raise NotImplementedError

    def getDistanceArray(self,):
        return np.reshape(self.distances, self.distances.size)

    def getMinVal(self):
        return np.min(self.distances)

    def getMaxVal(self):
        return np.max(self.distances)

    def getInterval(self):
        return self.getMinVal(), self.getMaxVal()
