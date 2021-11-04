import vtk
import torch
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as vtk2np


class TensorGenerator:
    def __init__(self):
        self.PATH = ""
        self.reader = None
        self.IJK = [0] * 3
        self.data = None
        self.type = None

    def set_path(self, path: str):
        self.PATH = path

    def set_type(self, type: str):
        self.type = type

    def update(self):
        self.reader = self._vtk_reader()
        self.data = self.reader.GetOutput()
        if type(self.data) is vtk.vtkMultiBlockDataSet:
            if self.data.GetBlock(0).GetPointData().GetNumberOfArrays() != 0:
                p2c = vtk.vtkPointDataToCellData()
                p2c.SetInputData(self.data.GetBlock(0))
                p2c.SetProcessAllArrays(True)
                p2c.PassPointDataOff()
                p2c.Update()
                self.data = p2c.GetOutput()
        else:
            if self.data.GetPointData().GetNumberOfArrays() != 0:
                p2c = vtk.vtkPointDataToCellData()
                p2c.SetInputData(self.data)
                p2c.SetProcessAllArrays(True)
                p2c.PassPointDataOff()
                p2c.Update()
                self.data = p2c.GetOutput()
        EXTENT = self.data.GetExtent()
        self.IJK[0] = EXTENT[1] - EXTENT[0]
        self.IJK[1] = EXTENT[3] - EXTENT[2]
        self.IJK[2] = EXTENT[5] - EXTENT[4]

    def _vtk_reader(self):
        if self.PATH.endswith('.vti'):
            reader = vtk.vtkXMLImageDataReader()
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

    def get_numpy_array(self, id:int) -> (np.ndarray, int):
        if not (0 <= id < self.data.GetCellData().GetNumberOfArrays()):
            ex = Exception("Error: ArrIndex out of bound.")
            raise ex

        IJKInArray = self.IJK
        IJKInArray.reverse()

        cdata = self.data.GetCellData().GetArray(id)
        tuples = cdata.GetNumberOfComponents()
        array = vtk2np(cdata).reshape(IJKInArray)
        return array, tuples

    def get_numpy_arrays(self) -> np.ndarray:
        if self.data is None:
            ex = Exception("Error: Data haven't been updated")
            raise ex

        if self.IS_UNIFORM_GRID is not True:
            ex = Exception("Error: Dataset is not uniform grid")
            raise ex

        '''
        Origin IJK is XYZ order, but in array, IJK should in ZYX order
        '''
        IJKInArray = self.IJK
        IJKInArray.reverse()

        # the 3d matrix, will be transform to arrays later
        # arrays = np.ndarray(shape=[cells.GetNumberOfArrays()] + IJKInArray)
        ''''''
        components = 0
        for i in range(self.data.GetCellData().GetNumberOfArrays()):
            components += self.data.GetCellData().GetArray(i).GetNumberOfComponents()
        arrays = np.ndarray([components] + IJKInArray)
        ''''''
        """
        if arrId < 0 or arrId >= cells.GetNumberOfArrays():
            ex = Exception("Error: ArrIndex out of bound.")
            raise ex
        """
        for arrId in range(self.data.GetCellData().GetNumberOfArrays()):
            cdata = self.data.GetCellData().GetArray(arrId)
            for components in range(1, cdata.GetNumberOfComponents()):
                arrays[cdata + components - 1] = vtk2np(cdata).reshape(IJKInArray)

        return arrays
        """
        cdata = cells.GetArray(arrId)
        array_type = cells.GetAttributeTypeAsString(arrId)
        if array_type == 'Scalars':
            arrays = vtk2np(cdata).reshape(IJKInArray)
        elif array_type == 'Vectors':
            arrays = np.linalg.norm(
                x=vtk2np(cdata).astype('float64'),
                ord=2, axis=1, keepdims=False,
            ).astype('float32').reshape(IJKInArray)
        else:
            ex = Exception("Error: Unsupported array type")
            raise ex

        return arrays
        """

    def generate_tensor(self):
        tensor = self.get_numpy_arrays()
        t = torch.from_numpy(tensor)

        return t

    def getOrigin(self):
        origin = [0.0] * 3
        try:
            origin = self.data.GetOrigin()
        except Exception:
            pass
        return origin

    def getSpacing(self):
        spacing = [1.0] * 3
        try:
            spacing = self.data.GetSpacing()
        except Exception:
            pass
        return spacing

    def getDimensions(self):
        return self.data.GetDimensions()
