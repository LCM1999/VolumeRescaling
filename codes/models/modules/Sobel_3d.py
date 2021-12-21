import torch
import torch.nn.functional as F


def sobel_edge_3d(inputTensor):
    sobel1 = torch.tensor([1, 0, -1], dtype=torch.float)
    sobel2 = torch.tensor([1, 2, 1], dtype=torch.float)

    sobel1xyz = [sobel1, sobel1, sobel1]
    sobel2xyz = [sobel2, sobel2, sobel2]

    for xyz in range(3):
        newShape = [1, 1, 1, 1, 1]
        newShape[xyz + 2] = 3
        sobel1xyz[xyz] = torch.reshape(sobel1, newShape)
        sobel2xyz[xyz] = torch.reshape(sobel2, newShape)

    output_x = F.conv3d(input=inputTensor,
                        weight=sobel1xyz[0],
                        stride=1,
                        padding=1)
    output_x = F.conv3d(input=output_x,
                        weight=sobel2xyz[1],
                        stride=1,
                        padding=1)
    output_x = F.conv3d(input=output_x,
                        weight=sobel2xyz[2],
                        stride=1,
                        padding=1)

    output_y = F.conv3d(input=inputTensor,
                        weight=sobel1xyz[1],
                        stride=1,
                        padding=1)
    output_y = F.conv3d(input=output_y,
                        weight=sobel2xyz[0],
                        stride=1,
                        padding=1)
    output_y = F.conv3d(input=output_y,
                        weight=sobel2xyz[2],
                        stride=1,
                        padding=1)

    output_z = F.conv3d(input=inputTensor,
                        weight=sobel1xyz[2],
                        stride=1,
                        padding=1)
    output_z = F.conv3d(input=output_z,
                        weight=sobel2xyz[0],
                        stride=1,
                        padding=1)
    output_z = F.conv3d(input=output_z,
                        weight=sobel2xyz[1],
                        stride=1,
                        padding=1)

    return torch.cat([output_x, output_y, output_z], 1)


if __name__ == '__main__':
    import codes.data.tensor_generator as Reader
    import codes.data.tensor_writer as Writer
    import numpy as np

    reader = Reader.TensorGenerator()
    # reader.set_path("E:\\TestYourCode\\test_o.vti")
    reader.set_path("E:\\pvOutput\\Ocean500g\\TLSI_000.vtk")
    reader.update()
    arr, tuples = reader.get_tensor_by_id(index=1)
    arr = torch.as_tensor(arr, dtype=torch.float)
    print(arr.numpy().shape)
    gradient = sobel_edge_3d(arr)
    print(int(torch.norm(input=gradient, p=0, dtype=torch.double).numpy()))
    gradient = gradient.numpy()
    print(gradient.shape)

    print(len(np.where(np.abs(gradient) > 0.0)[0]))
"""
writer = Writer.TensorWriter(filename="sobel",
                                 origin=reader.getOrigin(),
                                 spacing=reader.getSpacing(),
                                 dimensions=[x + 1 for x in gradient.shape[2:]])
    writer.append_data_channels(data=gradient,
                                name='gradient_pressure',
                                channel_index=1)
    writer.write()
"""

