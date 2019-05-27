# from PySide2 import QtWidgets
# from ui import main
# import sys
# import os
import numpy as np
from mesh_vox import read_and_reshape_stl, voxelize
from LBM import preprocessing
from LBM import lbm
import multiprocessing


class MyApp:
    def __init__(self):

        self.model_fileName = ''
        self.output_dir = ''

        self.resolution = 100
        self.outer_height = 10
        self.lb_unit = 0.05

        self.max_u_lb = 0.05

        self.dia_u_p = 0.01
        self.out_u_p = 0.01

        self.interval = 0.01
        self.duration = 1
        self.frequency = 300
        self.outer_flow_direction = 'no flow'
        self.Nu_P = 0.002
        self.dia_delay = 0

    def voxelize(self):

        resolution = int(self.resolution)
        mesh, self.bounding_box = read_and_reshape_stl(self.model_fileName, resolution)

        process_number = int(multiprocessing.cpu_count()) * 2
        self.voxels = voxelize(mesh, self.bounding_box, process_number)

    def generate_final_model(self):

        h = round(self.outer_height / self.lb_unit)

        self.voxels = np.delete(self.voxels, np.s_[self.bounding_box[1]:], 0)

        outer = np.zeros((h, self.bounding_box[0], self.bounding_box[2]), dtype=np.int8)

        if self.outer_flow_direction == 'no flow':
            outer[:, 0, :] = 1
            outer[:, -1, :] = 1
        elif self.outer_flow_direction == 'left to right':
            outer[:, 0, :] = 3
            outer[:, -1, :] = 1
        elif self.outer_flow_direction == 'right to left':
            outer[:, 0, :] = 1
            outer[:, -1, :] = 4

        outer[-1, :, :] = 1

        outer[:, :, 0] = 1

        outer[:, :, -1] = 1

        for i in range(self.bounding_box[0]):

            ff = 1
            bf = 1

            ffl = 1
            bfl = 1

            for j in range(self.bounding_box[2]):

                # if ff == 1 and self.voxels[-1, i, j] > -1:
                #
                #     self.voxels[-1, i, j] = 1
                #     self.voxels[-2, i, j] = -3
                #
                # elif self.voxels[-1, i, j] < 0:
                #     ff = 0
                #
                # if bf == 1 and self.voxels[-1, i, -1 - j] > -1:
                #     self.voxels[-1, i, -1 - j] = 1
                #     self.voxels[-2, i, -1 - j] = -3
                #
                # elif self.voxels[-1, i, -1 - j] < 0:
                #     bf = 0

                if ffl == 1 and self.voxels[0, i, j] > -1:

                    self.voxels[0, i, j] = -1

                elif self.voxels[0, i, j] < 0:
                    ffl = 0

                if bfl == 1 and self.voxels[0, i, -1 - j] > -1:

                    self.voxels[0, i, -1 - j] = -1

                elif self.voxels[0, i, -1 - j] < 0:
                    bfl = 0

        x = self.voxels[0]

        x[x == 0] = 2

        self.voxels[0] = x

        self.voxels = np.append(self.voxels, outer, axis=0)

    # rotate model around x axis - direction
    def rotate_x_minus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (2, 1, 0)), 0)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

    # rotate model around x axis + direction
    def rotate_x_plus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 0), (2, 1, 0))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

    # rotate model around z axis - direction
    def rotate_z_minus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 1), (0, 2, 1))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

    # rotate model around z axis + direction
    def rotate_z_plus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (0, 2, 1)), 1)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

    # run simulation         set output folder as parameter
    def start_simulation(self, out_folder='result'):

        preprocessing.node_initialization(self.voxels, self.bounding_box[1])

        lu = self.lb_unit

        duration = self.duration
        max_u_lb = self.max_u_lb
        dia_u_p = self.dia_u_p
        out_u_p = self.out_u_p

        frequency = self.frequency

        interval = self.interval

        ts = lu * max_u_lb / abs(dia_u_p)

        if abs(out_u_p) / (lu / ts) > 0.1:
            print('External Flow Speed Exceed 0.1 u')

        Nulb = lu * lu / ts

        dia_u_lb = dia_u_p * ts / lu

        if self.outer_flow_direction == 'no flow':
            out_u_lb = 0
        else:
            out_u_lb = out_u_p * ts / lu

        print("Time Step: {} s".format(ts))

        tau_inv = 1.0 / ((self.Nu_P / Nulb) * 3.0 + 0.5)

        tau_inv = 1.0 / 0.6

        print("tau_inverse: {}".format(tau_inv))

        lbm.solve(duration, lu, dia_u_lb, out_u_lb, tau_inv, ts, self.voxels, interval, frequency,
                  self.bounding_box[1], self.dia_delay, out_folder)


if __name__ == '__main__':

    # create simulator object
    app = MyApp()

    # num of lattices along the longest dimension of the nozzle model
    app.resolution = 120
    # dx between lattice nodes in m
    app.lb_unit = 0.2 / 1000
    # height of external region in m
    app.outer_height = 20 / 1000

    # maximum physical diaphram velocity
    app.dia_u_p = 0.5
    # maximum physical side flow velocity
    app.out_u_p = 4
    # duration of simulation in s
    app.duration = 0.013
    # diaphragm oscillation frequency
    app.frequency = 300

    # data output interval
    app.interval = 0.00000925925
    # max LB velocity of diaphragm
    app.max_u_lb = 0.005
    # physical k viscosity
    app.Nu_P = 14.8 / 1000000
    # nozzle model file name
    app.model_fileName = 'part33.STL'

    app.outer_flow_direction = 'no flow'

    app.dia_delay = 0
    # turn STL model into an array of Lattice Domain

    app.voxelize()

    # add external region above the nozzle    set outlet boundary
    app.generate_final_model()
    # run simulation         set output folder path as parameter
    app.start_simulation('result')
