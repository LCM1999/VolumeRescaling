import numpy as np


def _findPointForIndex(idx, dims):
    z = idx % dims[0]
    y = (idx / dims[0]) % dims[1]
    x = idx / (dims[1] * dims[0])

    return z, y, x


def _findOutputIndex(pt, dims, scale):
    return int((pt[2] * ((dims[1] - scale) * (dims[0] - scale)) + pt[1] * (dims[0] - scale) + pt[0]) / scale)


def getFlatVolume(volume):
    return np.reshape(volume, volume.size)


def checkInside(lu, isovalue, data, dims, scale):
    dimZY = scale * dims[0] * dims[1]
    id1 = lu + scale
    id2 = lu + scale * dims[0]
    id3 = id1 + scale * dims[0]
    id4 = lu + dimZY
    id5 = id1 + dimZY
    id6 = id2 + dimZY
    id7 = id3 + dimZY

    above = False
    below = False

    ooo = data[lu]
    above = ooo > isovalue
    below = ooo < isovalue
    ioo = data[id1]
    above = above or ioo > isovalue
    below = below or ioo < isovalue
    oio = data[id2]
    above = above or oio > isovalue
    below = below or oio < isovalue
    ooi = data[id3]
    above = above or ooi > isovalue
    below = below or ooi < isovalue
    iio = data[id4]
    above = above or iio > isovalue
    below = below or iio < isovalue
    ioi = data[id5]
    above = above or ioi > isovalue
    below = below or ioi < isovalue
    oii = data[id6]
    above = above or oii > isovalue
    below = below or oii < isovalue
    iii = data[id7]
    above = above or iii > isovalue
    below = below or iii < isovalue
    return above and below


def _calculate_approximation_CPU(data, isovalue, dims, surfaceApprox, size, offset, scale):
    for i in range(size):
        idx = i
        idx *= scale
        if idx < size:
            lu = _findPointForIndex(idx, dims)
            if (lu[2] < (dims[2] - scale)) and \
                    (lu[1] < (dims[1] - scale)) and \
                    (lu[0] < (dims[0] - scale)):
                # print(_findOutputIndex(lu, dims, scale))
                if checkInside(idx, isovalue, data, dims, scale):
                    surfaceApprox.append(lu)
                # surfaceApprox[_findOutputIndex(lu, dims, scale)] = \
                #     lu if checkInside(idx, isovalue, data, dims, scale) else (-1, -1, -1)


def ApproximateIsosurfaces_CPU(h_data, curIsovalue, dims, scale):
    SIZE = dims[0] * dims[1] * dims[2]
    h_points = []
    _calculate_approximation_CPU(h_data, curIsovalue, dims, h_points, SIZE, 0, scale)
    # print(h_points.shape)
    # size = (dims[0] - scale) * (dims[1] - scale) * (dims[2] - scale)
    # h_points2 = []
    # for i in range(size):
    #     if h_points[i][2] >= 0:
    #         h_points2.append(h_points[i])
    # h_points, h_points2 = h_points2, h_points
    # del h_points2
    print(len(h_points))
    return h_points


def ConvertCellDataToPoints(tuples, cellData: np.ndarray, components=1):
    dims = cellData.shape
    dimZ = dims[0] + 1
    dimY = dims[1] + 1
    dimX = dims[2] + 1

    data_flat = np.reshape(cellData, (tuples, components))

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as vtk2np

    image = vtk.vtkImageData()
    image.SetDimensions((dimZ, dimY, dimX))

    value = vtk.vtkFloatArray()
    value.SetName("tmp")
    value.SetNumberOfComponents(components)
    value.SetNumberOfTuples(image.GetNumberOfCells())
    for i in range(image.GetNumberOfCells()):
        value.SetValue(i, data_flat[i])

    image.GetCellData().AddArray(value)

    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(image)
    c2p.Update()

    pdata = c2p.GetOutput().GetPointData().GetArray(0)
    pointTuples = pdata.GetNumberOfComponents()
    pointArray = vtk2np(pdata).reshape((dimZ, dimY, dimX))

    return pointArray, pointTuples


