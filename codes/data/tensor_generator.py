import vtk
import torch
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as vtk2np


class TensorGenerator:
    def __init__(self):
        self.PATH = ""
        self.reader = None
        self.IJK = [0] * 3
        self.DIMS = None
        self.data = None
        self.type = None

    def set_path(self, path: str):
        self.PATH = path

    def set_type(self, type: str):
        self.type = type

    def update(self):
        self.reader = self._vtk_reader()
        self.data = self.reader.GetOutput()

        self.DIMS = self.data.GetDimensions()
        EXTENT = self.data.GetExtent()
        self.IJK[0] = EXTENT[1] - EXTENT[0]
        self.IJK[1] = EXTENT[3] - EXTENT[2]
        self.IJK[2] = EXTENT[5] - EXTENT[4]

        if self.type == 'point':
            self.data = self.reader.GetOutput().GetPointData()
        elif self.type == 'cell':
            self.data = self.reader.GetOutput().GetCellData()
        else:
            raise Exception("Unsupport type {}".format(self.type))
            # if type(self.data) is vtk.vtkMultiBlockDataSet:
            #     if self.data.GetBlock(0).GetPointData().GetNumberOfArrays() != 0:
            #         p2c = vtk.vtkPointDataToCellData()
            #         p2c.SetInputData(self.data.GetBlock(0))
            #         p2c.SetProcessAllArrays(True)
            #         p2c.PassPointDataOff()
            #         p2c.Update()
            #         self.data = p2c.GetOutput()
            # else:
            #     if self.data.GetPointData().GetNumberOfArrays() != 0:
            #         p2c = vtk.vtkPointDataToCellData()
            #         p2c.SetInputData(self.data)
            #         p2c.SetProcessAllArrays(True)
            #         p2c.PassPointDataOff()
            #         p2c.Update()
            #         self.data = p2c.GetOutput()

    def _vtk_reader(self):
        if self.PATH.endswith('.vti'):
            reader = vtk.vtkXMLImageDataReader()
        elif self.PATH.endswith('.vtk'):
            reader = vtk.vtkDataSetReader()
        elif self.PATH.endswith('.dat'):
            reader = vtk.vtkTecplotReader()
        else:
            ex = Exception("Error: Unsupported format")
            raise ex
        reader.SetFileName(self.PATH)
        try:
            reader.Update()
        except Exception as e:
            print(e)

        return reader

    def get_array_by_id(self, index: int) -> (np.ndarray, int):
        if not (0 <= index < self.data.GetNumberOfArrays()):
            ex = Exception("Error: ArrIndex out of bound.")
            raise ex

        if self.type == 'point':
            shape = list(self.DIMS)
        elif self.type == 'cell':
            shape = self.IJK
        else:
            raise Exception("Unsupport type {}".format(self.type))
        shape.reverse()

        data = self.data.GetArray(index)
        tuples = data.GetNumberOfComponents()
        array = vtk2np(data).reshape(shape)

        return array, tuples

    # def get_points_array_by_id(self, index: int) -> (np.ndarray, int):
    #     if not (0 <= index < self.data.GetCellData().GetNumberOfArrays()):
    #         ex = Exception("Error: ArrIndex out of bound.")
    #         raise ex
    #
    #     IJKInArray = self.IJK
    #     IJKInArray.reverse()
    #
    #     c2p = vtk.vtkCellDataToPointData()
    #     c2p.SetInputData(self.data)
    #     c2p.Update()
    #
    #     pdata = c2p.GetOutput().GetPointData().GetArray(index)
    #     pointTuples = pdata.GetNumberOfComponents()
    #     pointArray = vtk2np(pdata).reshape(IJKInArray)
    #
    #     return pointArray, pointTuples

    def get_tensor_by_id(self, index: int) -> (torch.Tensor, int):
        array, tuples = self.get_array_by_id(index)
        return torch.reshape(
            torch.from_numpy(array),
            [1, 1, array.shape[0], array.shape[1], array.shape[2]]
        ), tuples


