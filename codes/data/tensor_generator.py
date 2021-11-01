import vtk
import torch
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as vtk2np


class TensorGenerator:
    def __init__(self):
        self.PATH = ""
        self.reader = None
        self.GLOBAL_BOUNDS = None
        self.GLOBAL_XYZ = [0] * 3
        self.EXTENT = None
        self.CELLS_NUM = None
        self.IJK = [0] * 3
        self.IS_UNIFORM_GRID = False
        self.data = None

    def set_path(self, path: str):
        self.PATH = path

    def update(self):
        if len(self.PATH) == 0:
            ex = Exception("Error: Path haven't been set")
            raise ex

        self.reader = self._vtk_reader()
        self.data = self.reader.GetOutput()
        self.GLOBAL_BOUNDS = self.data.GetBounds()
        self.EXTENT = self.data.GetExtent()
        self.CELLS_NUM = self.data.GetNumberOfCells()
        self.GLOBAL_XYZ = [0] * 3
        self.IJK = [0] * 3
        self.IS_UNIFORM_GRID = self._is_uniform_grid()

    def _calculate_global_cells(self):
        self.GLOBAL_XYZ[0] = round(self.GLOBAL_BOUNDS[1]) - round(self.GLOBAL_BOUNDS[0])
        self.GLOBAL_XYZ[1] = round(self.GLOBAL_BOUNDS[3]) - round(self.GLOBAL_BOUNDS[2])
        self.GLOBAL_XYZ[2] = round(self.GLOBAL_BOUNDS[5]) - round(self.GLOBAL_BOUNDS[4])
        return self.GLOBAL_XYZ[0] * self.GLOBAL_XYZ[1] * self.GLOBAL_XYZ[2]

    def _is_uniform_grid(self):
        if (self._calculate_global_cells() / self.data.GetMaxCellSize()) == self.CELLS_NUM:
            self.IJK[0] = self.EXTENT[1] - self.EXTENT[0]
            self.IJK[1] = self.EXTENT[3] - self.EXTENT[2]
            self.IJK[2] = self.EXTENT[5] - self.EXTENT[4]
            return True
        else:
            return False

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
        return self.data.GetOrigin()

    def getSpacing(self):
        return self.data.GetSpacing()

    def getDimensions(self):
        return self.data.GetDimensions()
