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
    reader.set_path("E:\\JHTDB\\isotropic1024coarse\\p\\isotropic1024coarse_p_128_0.vti")
    reader.update()
    arr, tuples = reader.get_tensor_by_id(index=0)
    (batch, channel, W, H, D) = arr.size()
    gradient_o = sobel_edge_3d(arr)
    sizeO = gradient_o.numel()

    scale = 2
    origin_size = (batch, channel, W, H, D)
    down_size = (batch, channel, W / 2, H / 2, D / 2)

    downsampled = F.interpolate(input=arr, scale_factor=(1 / 2))
    gradient_d = sobel_edge_3d(downsampled)
    sizeD = gradient_d.numel()

    upsampled = F.interpolate(input=downsampled, size=(W, H, D), mode="area")
    gradient_u = sobel_edge_3d(upsampled)
    sizeU = gradient_u.numel()

    g_o = int(torch.norm(input=gradient_o, p=0, dtype=torch.double).numpy())
    g_d = int(torch.norm(input=gradient_d, p=0, dtype=torch.double).numpy())
    g_u = int(torch.norm(input=gradient_u, p=0, dtype=torch.double).numpy())

    g_o_n = int(torch.norm(input=gradient_o, p=0, dtype=torch.double).numpy()) / gradient_o.numel()
    g_d_n = int(torch.norm(input=gradient_d, p=0, dtype=torch.double).numpy()) / gradient_d.numel()
    g_u_n = int(torch.norm(input=gradient_u, p=0, dtype=torch.double).numpy()) / gradient_u.numel()

    # print("Gradient origin: {}, down: {}, up: {}".format(g_o, g_d, g_u))
    print("Normalize Gradient origin: {}, down: {}, up: {}".format(g_o_n, g_d_n, g_u_n))
    print("Gradient GTLR: {}".format(abs(scale ** 3 * g_d - g_o) / (scale ** 3 * sizeD - sizeO)))
    print("Gradient GTHR: {}".format(abs(g_u_n - g_o_n)))

    # writer = Writer.TensorWriter(filename="sobel",
    #                              origin=reader.getOrigin(),
    #                              spacing=reader.getSpacing(),
    #                              dimensions=[x + 1 for x in gradient.shape[2:]])
    # writer.append_data_channels(data=gradient,
    #                             name='gradient_pressure',
    #                             channel_index=1)
    # writer.write()
