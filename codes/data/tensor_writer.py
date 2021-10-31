import vtk
import numpy as np


class TensorWriter:
    def __init__(self, spacing, filename, dimensions, origin=(0, 0, 0)):
        self.writer = vtk.vtkXMLImageDataWriter()
        self.writer.SetFileName(filename + ".vti")
        self.grid = vtk.vtkImageData()
        self.grid.SetOrigin(origin)
        self.grid.SetSpacing(spacing)
        self.grid.SetDimensions(dimensions)

    def append_data(self, data: np.ndarray, name, components=1) -> int:
        data_flat = np.reshape(data, data.size)

        value = vtk.vtkFloatArray()
        value.SetName(name)
        value.SetNumberOfComponents(components)
        value.SetNumberOfTuples(self.grid.GetNumberOfCells())
        for i in range(self.grid.GetNumberOfCells()):
            value.SetValue(i, data_flat[i])

        self.grid.GetCellData().AddArray(value)
        return self.grid.GetCellData().GetNumberOfArrays()

    def write(self):
        if self.grid.GetCellData().GetNumberOfArrays() <= 0:
            ex = Exception("No data to Write")
            raise ex
        self.writer.SetInputData(self.grid)
        self.writer.Write()
