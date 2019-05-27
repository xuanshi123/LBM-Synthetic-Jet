import numpy as np
import math
from numba import cuda, njit, jit, prange, float32, float64, int32
import time
import multiprocessing
import concurrent.futures as cf
import threading
from LBM import out_result

e = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
              [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, -1, 0], [1, -1, 0],
              [-1, 1, 0], [0, 1, 1], [0, -1, -1], [1, 0, 1], [-1, 0, -1],
              [0, -1, 1], [0, 1, -1], [-1, 0, 1], [1, 0, -1]])

w = np.array([1.0 / 3,
              1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])

wd = w * 2

opp = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])


# cuda function to initialize f array
@cuda.jit('void(float32[:,:,:], int8[:,:,:], int32)')
def initialization_kernel(cur, voxels, h):
    dw = cuda.const.array_like(w)
    x, y = cuda.grid(2)

    if x < cur.shape[0] and y < cur.shape[1]:
        if voxels[h, x, y] >= 0:
            for i in range(19):
                cur[x, y, i] = dw[i]


# cuda function to calculate f after collision step
@cuda.jit('void(float32[:,:,:], float32, float32, int32)')
def collision_kernel(ftemp, tau_inv, v, direction):

    dw = cuda.const.array_like(w)
    x, y = cuda.grid(2)

    if x < ftemp.shape[0] and y < ftemp.shape[1] and ftemp[x, y, 0] > 0:

        # if processing diaphragm layer  use Zhou He to determine unknowns for fixed velocity boundary
        if direction == 2:

            rv = (ftemp[x,y,0] + ftemp[x,y,1] + ftemp[x,y,2] + ftemp[x,y,5] + ftemp[x,y,6] + ftemp[x,y,11]
                   + ftemp[x,y,12] + ftemp[x, y, 15] + ftemp[x,y,16] + 2 * (ftemp[x,y,4] + ftemp[x,y,8] + ftemp[x,y,10]
                                                              + ftemp[x,y,14] + ftemp[x,y,17])) / (1 - v) * v

            ftemp[x, y, 3] = ftemp[x, y, 4] + rv / 3

            diff_2_1 = ftemp[x, y, 2] - ftemp[x, y, 1]

            diff_5_6 = ftemp[x, y, 5] - ftemp[x, y, 6]

            diff_15_16 = ftemp[x, y, 15] - ftemp[x, y, 16]

            diff_12_11 = ftemp[x, y, 12] - ftemp[x, y, 11]

            ftemp[x, y, 7] = ftemp[x, y, 8] + rv / 6 + (diff_2_1 + diff_15_16 + diff_12_11) / 2

            ftemp[x, y, 9] = ftemp[x, y, 10] + rv / 6 - (diff_2_1 + diff_15_16 + diff_12_11) / 2

            ftemp[x, y, 13] = ftemp[x, y, 14] + rv / 6 - (diff_5_6 + diff_15_16 - diff_12_11) / 2

            ftemp[x, y, 18] = ftemp[x, y, 17] + rv / 6 + (diff_5_6 + diff_15_16 - diff_12_11) / 2

        # if processing left to right external flow  use Zhou He to determine unknowns for fixed velocity boundary
        elif x == 0 and direction == 3:

            ru = (ftemp[x,y,0] + ftemp[x,y,3] + ftemp[x,y,4] + ftemp[x,y,5] + ftemp[x,y,6] + ftemp[x,y,13] + ftemp[x,y,14]
                   + ftemp[x,y,17] + ftemp[x,y,18] + 2 * (ftemp[x, y, 2] + ftemp[x, y, 8] + ftemp[x, y, 9] + ftemp[x,y,12]
                                                            + ftemp[x,y,15])) / (1 - v) * v

            ftemp[x, y, 1] = ftemp[x, y, 2] + ru / 3

            diff_3_4 = ftemp[x, y, 3] - ftemp[x, y, 4]

            diff_13_14 = ftemp[x, y, 13] - ftemp[x, y, 14]

            diff_18_17 = ftemp[x, y, 18] - ftemp[x, y, 17]

            diff_5_6 = ftemp[x, y, 5] - ftemp[x, y, 6]

            ftemp[x, y, 7] = ftemp[x, y, 8] + ru / 6 - (diff_3_4 + diff_13_14 + diff_18_17) / 2

            ftemp[x, y, 10] = ftemp[x, y, 9] + ru / 6 + (diff_3_4 + diff_13_14 + diff_18_17) / 2

            ftemp[x, y, 11] = ftemp[x, y, 12] + ru / 6 - (diff_5_6 + diff_13_14 - diff_18_17) / 2

            ftemp[x, y, 16] = ftemp[x,y,15] + ru / 6 + (diff_5_6 + diff_13_14 - diff_18_17) / 2

        # if processing right to left external flow  use Zhou He to determine unknowns for fixed velocity boundary
        elif x == ftemp.shape[0] - 1 and direction == 4:

            ru = (ftemp[x,y,0] + ftemp[x,y,3] + ftemp[x,y,4] + ftemp[x,y,5] + ftemp[x,y,6] + ftemp[x,y,13] + ftemp[x,y,14]
                   + ftemp[x,y,17] + ftemp[x,y,18] + 2 * (ftemp[x,y,1] + ftemp[x,y,7] + ftemp[x,y,10]
                                                          + ftemp[x,y,11] + ftemp[x,y,16])) / (1 + v) * v

            ftemp[x, y, 2] = ftemp[x, y, 1] - ru / 3

            diff_3_4 = ftemp[x, y, 3] - ftemp[x, y, 4]

            diff_13_14 = ftemp[x, y, 13] - ftemp[x, y, 14]

            diff_18_17 = ftemp[x, y, 18] - ftemp[x, y, 17]

            diff_5_6 = ftemp[x, y, 5] - ftemp[x, y, 6]

            ftemp[x, y, 8] = ftemp[x, y, 7] - ru / 6 + (diff_3_4 + diff_13_14 + diff_18_17) / 2

            ftemp[x, y, 9] = ftemp[x, y, 10] - ru / 6 - (diff_3_4 + diff_13_14 + diff_18_17) / 2

            ftemp[x, y, 12] = ftemp[x, y, 11] - ru / 6 + (diff_5_6 + diff_13_14 - diff_18_17) / 2

            ftemp[x, y, 15] = ftemp[x, y, 16] - ru / 6 - (diff_5_6 + diff_13_14 - diff_18_17) / 2

        rho = 0

        for i in range(19):
            rho += ftemp[x, y, i]

        vx = (ftemp[x, y, 1] - ftemp[x, y, 2] + ftemp[x, y, 7] - ftemp[x, y, 8] + ftemp[x, y, 10] - ftemp[x, y, 9] +
              ftemp[x, y, 11] - ftemp[x, y, 12] + ftemp[x, y, 16] - ftemp[x, y, 15]) / rho

        vy = (ftemp[x, y, 3] - ftemp[x, y, 4] + ftemp[x, y, 7] - ftemp[x, y, 8] + ftemp[x, y, 9] - ftemp[x, y, 10] +
              ftemp[x, y, 13] - ftemp[x, y, 14] + ftemp[x, y, 18] - ftemp[x, y, 17]) / rho

        vz = (ftemp[x, y, 5] - ftemp[x, y, 6] + ftemp[x, y, 11] - ftemp[x, y, 12] + ftemp[x, y, 13] - ftemp[x, y, 14]
              + ftemp[x, y, 15] - ftemp[x, y, 16] + ftemp[x, y, 17] - ftemp[x, y, 18]) / rho

        # calculate feq and f after collision

        square = 1.5 * (vx * vx + vy * vy + vz * vz)

        ftemp[x, y, 0] += (dw[0] * rho * (1 - square) - ftemp[x, y, 0]) * tau_inv

        feq1 = dw[1] * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - square)

        ftemp[x, y, 1] += (feq1 - ftemp[x, y, 1]) * tau_inv

        ftemp[x, y, 2] += (feq1 - 6.0 * dw[1] * rho * vx - ftemp[x, y, 2]) * tau_inv

        feq3 = dw[3] * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - square)

        ftemp[x, y, 3] += (feq3 - ftemp[x, y, 3]) * tau_inv

        ftemp[x, y, 4] += (feq3 - 6.0 * dw[3] * rho * vy - ftemp[x, y, 4]) * tau_inv

        feq5 = dw[5] * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - square)

        ftemp[x, y, 5] += (feq5 - ftemp[x, y, 5]) * tau_inv

        ftemp[x, y, 6] += (feq5 - 6.0 * dw[5] * rho * vz - ftemp[x, y, 6]) * tau_inv

        sum = vx + vy

        vxy = 2.0 * vx * vy

        vxy2 = vx * vx + vy * vy

        feq7 = dw[7] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 + vxy) - square)

        ftemp[x, y, 7] += (feq7 - ftemp[x, y, 7]) * tau_inv
        ftemp[x, y, 8] += (feq7 - 6.0 * dw[7] * rho * sum - ftemp[x, y, 8]) * tau_inv

        sum = vy - vx

        feq9 = dw[9] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 - vxy) - square)

        ftemp[x, y, 9] += (feq9 - ftemp[x, y, 9]) * tau_inv
        ftemp[x, y, 10] += (feq9 - 6.0 * dw[9] * rho * sum - ftemp[x, y, 10]) * tau_inv

        sum = vx + vz

        vxz = 2.0 * vx * vz

        vxz2 = vx * vx + vz * vz

        feq11 = dw[11] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 + vxz) - square)

        ftemp[x, y, 11] += (feq11 - ftemp[x, y, 11]) * tau_inv
        ftemp[x, y, 12] += (feq11 - 6.0 * dw[11] * rho * sum - ftemp[x, y, 12]) * tau_inv

        sum = vy + vz

        vyz = 2.0 * vy * vz

        vyz2 = vy * vy + vz * vz

        feq13 = dw[13] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz + vyz2) - square)

        ftemp[x, y, 13] += (feq13 - ftemp[x, y, 13]) * tau_inv
        ftemp[x, y, 14] += (feq13 - 6.0 * dw[13] * rho * sum - ftemp[x, y, 14]) * tau_inv

        sum = vz - vx

        feq15 = dw[15] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 - vxz) - square)

        ftemp[x, y, 15] += (feq15 - ftemp[x, y, 15]) * tau_inv
        ftemp[x, y, 16] += (feq15 - 6.0 * dw[15] * rho * sum - ftemp[x, y, 16]) * tau_inv

        sum = vz - vy

        feq17 = dw[17] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz2 - vyz) - square)

        ftemp[x, y, 17] += (feq17 - ftemp[x, y, 17]) * tau_inv
        ftemp[x, y, 18] += (feq17 - 6.0 * dw[17] * rho * sum - ftemp[x, y, 18]) * tau_inv


# thread function to handle collision step
def collision_processing(ftable, dia_u_lb, out_u_lb, total_height,
                   blockspergrid, threadsperblock, tau_inv, num_device, gpu_id, ext_dir):

    # select gpu according to assigned id
    cuda.select_device(gpu_id)

    # create workflow stream to handle asynchronous data transfer and function execution
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    start = gpu_id * total_height // num_device
    end = (gpu_id + 1) * total_height // num_device

    # process layers along y axis

    for i in range(start, end):

        if i == 0:
            v = dia_u_lb
            direction = 2
        else:
            v = out_u_lb
            direction = ext_dir

        if i == start:
            # transfer the first layer to device memory
            dcur = cuda.to_device(ftable[i], stream=stream1)

        else:
            dcur = dnext
            stream2.synchronize()

        collision_kernel[blockspergrid, threadsperblock, stream1](dcur, tau_inv, v, direction)

        if i < end - 1:
            dnext = cuda.to_device(ftable[i+1], stream=stream2)

        stream1.synchronize()
        dcur.copy_to_host(ftable[i], stream=stream3)

    stream3.synchronize()


# thread function to handle streaming step
@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:])')
def propagate_kernel(dtemp, dcur, dabove, dbelow):

    # set constant array of direction vectors
    de = cuda.const.array_like(e)
    # set constant array of opposite direction indices
    dopp = cuda.const.array_like(opp)
    # obtain position in grid
    x, y = cuda.grid(2)

    # if within domain and  not solid node
    if x < dcur.shape[0] and y < dcur.shape[1]:

        if dcur[x, y, 0] > 0:

            dtemp[x, y, 0] = dcur[x, y, 0]

            for i in range(1, 19):

                # obtain original position of f distribution
                xi = x - de[i, 1]
                yi = y - de[i, 2]

                # get the closest node if origin outside domain
                if xi == -1:
                    xi = 0
                elif xi == dcur.shape[0]:
                    xi = dcur.shape[0] - 1

                if yi == -1:
                    yi = 0
                elif yi == dcur.shape[1]:
                    yi = dcur.shape[1] - 1

                # if from current layer
                if de[i, 0] == 0:

                    # halfway bounceback
                    if dcur[xi, yi, i] == 0:
                        j = dopp[i]
                        dtemp[x, y, i] = dcur[x, y, j]
                    else:
                        dtemp[x, y, i] = dcur[xi, yi, i]

                # if from layer below
                elif de[i, 0] > 0:

                    # halfway bounceback
                    if dbelow[xi, yi, i] == 0:
                        j = dopp[i]
                        dtemp[x, y, i] = dcur[x, y, j]
                    else:
                        dtemp[x, y, i] = dbelow[xi, yi, i]
                # if from layer above
                elif de[i, 0] < 0:

                    # halfway bounceback
                    if dabove[xi, yi, i] == 0:
                        j = dopp[i]
                        dtemp[x, y, i] = dcur[x, y, j]
                    else:
                        dtemp[x, y, i] = dabove[xi, yi, i]

        else:
            for i in range(19):
                dtemp[x, y, i] = 0


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:,:])')
def propagate_kernel_top(dtemp, dcur, dbelow):

    de = cuda.const.array_like(e)
    dopp = cuda.const.array_like(opp)
    x, y = cuda.grid(2)

    if x < dcur.shape[0] and y < dcur.shape[1]:

        if dcur[x, y, 0] > 0:
            dtemp[x, y, 0] = dcur[x, y, 0]

            for i in range(1, 19):

                xi = x - de[i, 1]
                yi = y - de[i, 2]

                if xi == -1:
                    xi = 0
                elif xi == dcur.shape[0]:
                    xi = dcur.shape[0] - 1

                if yi == -1:
                    yi = 0
                elif yi == dcur.shape[1]:
                    yi = dcur.shape[1] - 1

                if de[i, 0] == 0:
                    if dcur[xi, yi, i] == 0:
                        j = dopp[i]
                        dtemp[x, y, i] = dcur[x, y, j]
                    else:
                        dtemp[x, y, i] = dcur[xi, yi, i]
                elif de[i, 0] > 0:
                    if dbelow[xi, yi, i] == 0:
                        j = dopp[i]
                        dtemp[x, y, i] = dcur[x, y, j]
                    else:
                        dtemp[x, y, i] = dbelow[xi, yi, i]
                elif de[i, 0] < 0:
                    dtemp[x, y, i] = dcur[xi, yi, i]

        else:
            for i in range(19):
                dtemp[x, y, i] = 0


# thread function to handle streaming step
def propagate(ftable, ftable_empty, height, blockspergrid, threadsperblock, gpu_id, num_device
              , dia_u_lb=0, out_u_lb=0, ext_dir=0, tau_inv=0):

    cuda.select_device(gpu_id)
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    start = gpu_id * height // num_device
    end = (gpu_id + 1) * height // num_device

    for i in range(start, end):

        dtemp = cuda.device_array_like(ftable_empty[0], stream=stream1)

        if i == 0:

            dabove = cuda.to_device(ftable[1], stream=stream1)
            dcur = cuda.to_device(ftable[0], stream=stream1)
            dbelow = dcur

        elif i == start:

            dbelow = cuda.to_device(ftable[i - 1], stream=stream1)
            dabove = cuda.to_device(ftable[i + 1], stream=stream1)
            dcur = cuda.to_device(ftable[i], stream=stream1)

        else:
            dbelow = dcur
            dcur = dabove
            dabove = dnext

        if i == height - 1:
            stream2.synchronize()
            propagate_kernel_top[blockspergrid, threadsperblock, stream1](dtemp, dcur, dbelow)

        else:
            stream2.synchronize()
            propagate_kernel[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dbelow)

        if i < height - 2:
            dnext = cuda.to_device(ftable[i + 2], stream=stream2)

        if i == 0:
            v = dia_u_lb
            direction = 2
        else:
            v = out_u_lb
            direction = ext_dir

        collision_kernel[blockspergrid, threadsperblock, stream1](dtemp, tau_inv, v, direction)

        stream1.synchronize()
        dtemp.copy_to_host(ftable_empty[i], stream=stream3)

    stream3.synchronize()


def initialization(voxels, ftable, blockspergrid, threadsperblock):

    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    dvoxels = cuda.to_device(voxels, stream=stream1)

    for i in range(voxels.shape[0]):

        if i == 0:
            dcur = cuda.to_device(ftable[i], stream=stream1)

        else:

            stream2.synchronize()
            dcur = dnext

        if i < voxels.shape[0] - 1:
            dnext = cuda.to_device(ftable[i + 1], stream=stream2)

        initialization_kernel[blockspergrid, threadsperblock, stream1](dcur, dvoxels, i)
        stream1.synchronize()
        dcur.copy_to_host(ftable[i], stream=stream3)

    stream3.synchronize()


def solve(duration, lu, dia_u_lb_max, out_u_lb, tau_inv, ts, voxels
          , interval, frequency, nozzle_height, dia_delay, out_folder):

    # create array to hold 19 f distributions of lattice domain
    ftable = np.zeros((voxels.shape[0], voxels.shape[1], voxels.shape[2], 19), np.float32)

    # create array to hold f population after streaming step
    ftable_empty = np.zeros((voxels.shape[0], voxels.shape[1], voxels.shape[2], 19), np.float32)

    # define external flow direction
    ext_dir = 0

    # set left to right direction
    if voxels[nozzle_height, 0, 1] == 3:
        ext_dir = 3

    # set right to left direction
    elif voxels[nozzle_height, -1, 1] == 4:
        ext_dir = 4

    # print(ext_dir)

    # define GPU setting   threads per block  and  blocks per grid   for 1 layer along y axis
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(voxels.shape[1] / threadsperblock[0])
    blockspergrid_z = math.ceil(voxels.shape[2] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_z)

    # initialize f distributions according to types of node
    initialization(voxels, ftable, blockspergrid, threadsperblock)

    # find number of GPU devices
    device_num = len(cuda.gpus)

    # initialize time and number of data output
    t = 0
    output_num = 0

    # calculate LB velocity of diaphragm
    dia_u_lb = 0 if t < dia_delay else dia_u_lb_max * np.sin(2 * np.pi * (t - dia_delay) * frequency)

    # create parallel thread pool
    with cf.ThreadPoolExecutor(max_workers=device_num) as executor:

        for i in range(device_num):
            # create thread to run collision step calculation for each GPU
            executor.submit(collision_processing, ftable, dia_u_lb, out_u_lb, voxels.shape[0],
                            blockspergrid, threadsperblock, tau_inv, device_num, i, ext_dir)

    while t < duration:

        # start = time.time()

        # output data result after specific interval
        if t >= output_num * interval:

            output = multiprocessing.Process(target=out_result.output_result, args=(lu, ts,
                            ftable[:, :, math.floor(voxels.shape[2] / 2), :], out_folder, t))
            output.start()
            output_num += 1

        # increment time by time step
        t += ts

        dia_u_lb = 0 if t < dia_delay else dia_u_lb_max * np.sin(2 * np.pi * (t - dia_delay) * frequency)

        # create parallel thread pool
        with cf.ThreadPoolExecutor(max_workers=device_num) as executor:

                for i in range(device_num):
                    # create thread to run streaming step and collision step at same time
                    executor.submit(propagate, ftable, ftable_empty, voxels.shape[0], blockspergrid,
                                    threadsperblock, i, device_num, dia_u_lb, out_u_lb, ext_dir, tau_inv)

        temp = ftable

        ftable = ftable_empty

        ftable_empty = temp

        # print("time spent for processing domain :", time.time() - start, "s")


