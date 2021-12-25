import vtk
import numpy as np


class TensorWriter:
    def __init__(self, filename, dimensions, spacing=(1, 1, 1), origin=(0, 0, 0)):
        self.writer = vtk.vtkXMLImageDataWriter()
        self.writer.SetFileName(filename + ".vti")
        self.grid = vtk.vtkImageData()
        self.grid.SetOrigin(origin)
        self.grid.SetSpacing(spacing)
        self.grid.SetDimensions(dimensions)
        self.tuples = self.grid.GetNumberOfCells()

    def append_data_tuple(self, data: np.ndarray, name, components=1, isPoint=True) -> int:
        if data.size != self.tuples:
            ex = Exception("data arr elements({}) != grid tuples number({})".format(data.size, self.tuples))
            raise ex
        data_flat = np.reshape(data, (self.tuples, components))

        value = vtk.vtkFloatArray()
        value.SetName(name)
        value.SetNumberOfComponents(components)
        if isPoint:
            value.SetNumberOfTuples(self.grid.GetNumberOfPoints())
            for i in range(self.grid.GetNumberOfPoints()):
                value.SetValue(i, data_flat[i])
            self.grid.GetPointData().AddArray(value)
            return self.grid.GetPointData().GetNumberOfArrays()
        else:
            value.SetNumberOfTuples(self.grid.GetNumberOfCells())
            for i in range(self.grid.GetNumberOfCells()):
                value.SetValue(i, data_flat[i])
            self.grid.GetCellData().AddArray(value)
            return self.grid.GetCellData().GetNumberOfArrays()

    def append_data_channels(self, data: np.ndarray, name, channel_index=0, isPoint=True) -> int:
        channels = data.shape[channel_index]
        if (data.size / channels) != self.tuples:
            ex = Exception("data arr elements({}) != grid tuples number({})".format((data.size / channels),
                                                                                    self.tuples))
            raise ex
        data_flats = np.reshape(data, (channels, self.tuples))

        value = vtk.vtkFloatArray()
        value.SetName(name)
        value.SetNumberOfComponents(channels)
        if isPoint:
            value.SetNumberOfTuples(self.grid.GetNumberOfPoints())
            for i in range(self.grid.GetNumberOfPoints()):
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
            self.grid.GetPointData().AddArray(value)
            return self.grid.GetPointData().GetNumberOfArrays()
        else:
            value.SetNumberOfTuples(self.grid.GetNumberOfCells())
            for i in range(self.grid.GetNumberOfCells()):
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
