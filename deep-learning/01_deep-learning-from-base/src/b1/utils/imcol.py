import numpy as np

def im2col(input_data, fh, fw, stride=1, pad=0): # (N C H W), (FH FW) -> (N*OH*OW C*FH*FW) 
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - fh)//stride + 1
    out_w = (W + 2*pad - fw)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, fh, fw, out_h, out_w))

    for y in range(fh):
        y_max = y + stride*out_h
        for x in range(fw):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # (N C FH FW OH OW) -> (N OH OW C FH FW)
    return col


def col2im(col, input_shape, fh, fw, stride=1, pad=0):# (N*OH*OW C*FH*FW) -> (N C H W), (FH FW)
    N, C, H, W = input_shape
    out_h = (H + 2*pad - fh)//stride + 1
    out_w = (W + 2*pad - fw)//stride + 1
    col = col.reshape(N, out_h, out_w, C, fh, fw).transpose(0, 3, 4, 5, 1, 2) # (N OH OW C FH FW) -> (N C FH FW OH OW)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(fh):
        y_max = y + stride*out_h
        for x in range(fw):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]