import vtk
import itk
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
        self.GENERATE_START = False
        self.GENERATE_END = False

    def update(self):
        if len(self.PATH) == 0:
            ex = Exception("Error: Path haven't been set")
            raise ex

        self.reader = self._vtk_reader()
        self.GLOBAL_XYZ = [0] * 3
        self.IJK = [0] * 3
        self.IS_UNIFORM_GRID = False
        self.data = self.reader.GetOutput()

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
        else:
            ex = Exception("Error: Unsupported format")
            raise ex
        reader.SetFileName(self.PATH)
        try:
            reader.Update()
        except Exception as e:
            print(e)

        return reader

    def get_numpy_array(self, arrId) -> np.ndarray:
        if self.data is None:
            ex = Exception("Error: Data haven't been updated")
            raise ex

        self.GLOBAL_BOUNDS = self.data.GetBounds()
        self.EXTENT = self.data.GetExtent()
        self.CELLS_NUM = self.data.GetNumberOfCells()
        self.IS_UNIFORM_GRID = self._is_uniform_grid()

        if self.IS_UNIFORM_GRID is not True:
            ex = Exception("Error: Dataset is not uniform grid")
            raise ex

        cells = self.data.GetCellData()
        '''
        Origin IJK is XYZ order, but in array, IJK should in ZYX order
        '''
        IJKInArray = self.IJK
        IJKInArray.reverse()
        # the 3d matrix, will be transform to tensor later
        # tensor = np.ndarray(shape=[cells.GetNumberOfArrays()] + IJKInArray)
        tensor = np.ndarray(IJKInArray)

        if arrId < 0 or arrId >= cells.GetNumberOfArrays():
            ex = Exception("Error: ArrIndex out of bound.")
            raise ex

        cdata = cells.GetArray(arrId)
        array_type = cells.GetAttributeTypeAsString(arrId)
        if array_type == 'Scalars':
            tensor = vtk2np(cdata).reshape(IJKInArray)
        elif array_type == 'Vectors':
            tensor = np.linalg.norm(
                x=vtk2np(cdata).astype('float64'),
                ord=2, axis=1, keepdims=False,
            ).astype('float32').reshape(IJKInArray)
        else:
            ex = Exception("Error: Unsupported array type")
            raise ex

        return tensor

        '''
        for i in range(0, cells.GetNumberOfArrays()):
            cdata = cells.GetArray(i)
            array_type = cells.GetAttributeTypeAsString(i)
            if array_type == 'Scalars':
                tensor[i] = vtk2np(cdata).reshape(IJKInArray)
            elif array_type == 'Vectors':
                tensor[i] = np.linalg.norm(
                    x=vtk2np(cdata).astype('float64'),
                    ord=2, axis=1, keepdims=False,
                ).astype('float32').reshape(IJKInArray)
            else:
                ex = Exception("Error: Unsupported array type")
                raise ex

        return tensor
        '''

    def generate_tensor(self):
        tensor = self.get_numpy_array()
        t = torch.from_numpy(tensor)

        return t

    def get_Origin(self):
        return self.data.GetOrigin()

    def get_Spacing(self):
        return self.data.GetSpacing()

    def get_Dimensions(self):
        return self.data.GetDimensions()
