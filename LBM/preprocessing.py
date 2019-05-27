import numpy as np
from numba import cuda
import math


@cuda.jit('void(int8[:,:], int8[:,:], int8[:,:])')
def stream_kernel(cur, above, below):

    x, y = cuda.grid(2)

    if 0 < x < cur.shape[0] - 1 and 0 < y < cur.shape[1] - 1:
        if cur[x, y] == -1:
            if cur[x+1, y] > -1 or cur[x, y+1] > -1 or cur[x+1, y+1] > -1 or cur[x-1, y] > -1 or cur[x, y-1] > -1 or\
                    cur[x-1, y-1] > -1 or cur[x+1, y-1] > -1 or cur[x-1, y+1] > -1 or above[x, y] > -1 or\
                    above[x+1, y] > -1 or above[x, y+1] > -1 or above[x+1, y-1] > -1 or\
                    above[x-1, y+1] > -1 or above[x+1, y+1] > -1 or above[x-1, y] > -1 or above[x, y-1] > -1 or\
                    above[x-1, y-1] > -1 or below[x, y] > -1 or below[x+1, y] > -1 or below[x+1, y-1] > -1 or\
                    below[x-1, y+1] > -1 or below[x, y+1] > -1 or below[x+1, y+1] > -1 or \
                    below[x-1, y] > -1 or below[x, y-1] > -1 or below[x-1, y-1] > -1:
                cur[x, y] = -2


@cuda.jit('void(int8[:,:])')
def stream_kernel_2(cur):

    x, y = cuda.grid(2)

    if 0 < x < cur.shape[0] - 1 and 0 < y < cur.shape[1] - 1:
        if cur[x, y] == -1:
            if cur[x+1, y] == -3 or cur[x, y+1] == -3 or cur[x+1, y+1] == -3 or cur[x-1, y] == -3 or cur[x, y-1] == -3\
                    or cur[x-1, y-1] == -3 or cur[x+1, y-1] == -3 or cur[x-1, y+1] == -3:
                cur[x, y] = -3


@cuda.jit('void(int8[:,:])')
def stream_kernel_3(cur):

    x, y = cuda.grid(2)

    if 0 <= x <= cur.shape[0] - 1 and 0 <= y <= cur.shape[1] - 1:
        if cur[x, y] == -1:
                cur[x, y] = -2


def node_initialization(node, height):

    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(node.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(node.shape[2] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for i in range(height):

        if i == 0:
            below = np.ones((node.shape[1], node.shape[2]), dtype=np.int8) * -1

            dbelow = cuda.to_device(below, stream=stream1)
            dabove = cuda.to_device(node[i + 1], stream=stream1)
            dcur = cuda.to_device(node[i], stream=stream1)

            dnext = cuda.to_device(node[i + 2], stream=stream2)
            stream_kernel[blockspergrid, threadsperblock, stream1](dcur, dabove, dbelow)

        else:

            if i == height - 1:

                dcur = dabove
                stream_kernel_3[blockspergrid, threadsperblock, stream1](dcur)

            else:

                dbelow = dcur
                dcur = dabove
                dabove = dnext

                if i < height - 2:
                    dnext = cuda.to_device(node[i + 2], stream=stream2)

                # elif i == height - 2:
                #     stream_kernel_2[blockspergrid, threadsperblock, stream1](dcur)

                stream_kernel[blockspergrid, threadsperblock, stream1](dcur, dabove, dbelow)

            stream3.synchronize()

        stream1.synchronize()
        dcur.copy_to_host(node[i], stream=stream3)
        stream2.synchronize()

    stream3.synchronize()




