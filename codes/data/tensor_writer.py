import vtk
import numpy as np


class TensorWriter:
    def __init__(self, filename, spacing, dimensions, origin=(0, 0, 0)):
        self.writer = vtk.vtkXMLImageDataWriter()
        self.writer.SetFileName(filename + ".vti")
        self.grid = vtk.vtkImageData()
        self.grid.SetOrigin(origin)
        self.grid.SetSpacing(spacing)
        self.grid.SetDimensions(dimensions)
        self.tuples = self.grid.GetNumberOfCells()

    def append_data_tuple(self, data: np.ndarray, name, components=1) -> int:
        if data.size != self.tuples:
            ex = Exception("data arr elements({}) != grid tuples number({})".format(data.size, self.tuples))
            raise ex
        data_flat = np.reshape(data, (self.tuples, components))

        value = vtk.vtkFloatArray()
        value.SetName(name)
        value.SetNumberOfComponents(components)
        value.SetNumberOfTuples(self.grid.GetNumberOfCells())
        for i in range(self.grid.GetNumberOfCells()):
            value.SetValue(i, data_flat[i])

        self.grid.GetCellData().AddArray(value)
        return self.grid.GetCellData().GetNumberOfArrays()

    def append_data_channels(self, data: np.ndarray, name, channel_index=0) -> int:
        channels = data.shape[channel_index]
        if (data.size / channels) != self.tuples:
            ex = Exception("data arr elements({}) != grid tuples number({})".format((data.size / channels),
                                                                                    self.tuples))
            raise ex
        data_flats = np.reshape(data, (channels, self.tuples))

        value = vtk.vtkFloatArray()
        value.SetName(name)
        value.SetNumberOfComponents(channels)
        value.SetNumberOfTuples(self.tuples)
        for i in range(self.tuples):
            if channels == 1:
                value.SetTuple1(i, data_flats[0][i])
            elif channels == 2:
                value.SetTuple2(i, data_flats[0][i],
                                data_flats[1][i])
            elif channels == 3:
                value.SetTuple3(i, data_flats[0][i],
                                data_flats[1][i],
                                data_flats[2][i])
            elif channels == 4:
                value.SetTuple4(i, data_flats[0][i],
                                data_flats[1][i],
                                data_flats[2][i],
                                data_flats[3][i])
            elif channels == 5:
                value.SetTuple5(i, data_flats[0][i],
                                data_flats[1][i],
                                data_flats[2][i],
                                data_flats[3][i],
                                data_flats[4][i])
            elif channels == 6:
                value.SetTuple6(i, data_flats[0][i],
                                data_flats[1][i],
                                data_flats[2][i],
                                data_flats[3][i],
                                data_flats[4][i],
                                data_flats[5][i])
            else:
                ex = Exception("Unsupport components {}".format(channels))

        self.grid.GetCellData().AddArray(value)
        return self.grid.GetCellData().GetNumberOfArrays()

    def write(self):
        if self.grid.GetCellData().GetNumberOfArrays() <= 0:
            ex = Exception("No data to Write")
            raise ex
        self.writer.SetInputData(self.grid)
        self.writer.Write()
