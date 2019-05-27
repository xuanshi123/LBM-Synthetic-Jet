from PySide2 import QtWidgets
from ui import main
import sys
import os
import numpy as np
from mesh_vox import read_and_reshape_stl, voxelize
from LBM import preprocessing
from LBM import lbm
import multiprocessing


class MyQtApp(main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__()
        self.setupUi(self)
        # self.update_graph()
        self.pushButton.clicked.connect(self.open_3d_model)
        self.pushButton_6.clicked.connect(self.voxelize)
        self.pushButton_2.clicked.connect(self.rotate_x_plus)
        self.pushButton_3.clicked.connect(self.rotate_x_minus)
        self.pushButton_4.clicked.connect(self.rotate_z_plus)
        self.pushButton_5.clicked.connect(self.rotate_z_minus)
        self.pushButton_7.clicked.connect(self.center_plot)
        # self.pushButton_10.clicked.connect(self.obtain_output_folder)

        self.pushButton_9.clicked.connect(self.start_simulation)

        self.pushButton_11.clicked.connect(self.display_time_step)

        self.spinBox_2.setProperty("value", int(multiprocessing.cpu_count()) * 2)

        self.spinBox.setProperty("value", 100)

        self.model_fileName = ''
        self.output_dir = ''

        self.i0 = np.array([])
        self.i1 = np.array([])
        self.i2 = np.array([])
        self.o1 = np.array([])
        self.o2 = np.array([])

        self.lineEdit.setText('0.1')
        self.lineEdit_2.setText('2')

    def display_time_step(self):

        lu = float(self.lineEdit.text())

        max_u_lb = float(self.lineEdit_5.text())

        dia_u_p = float(self.lineEdit_4.text())

        out_u_p = float(self.lineEdit_7.text())

        ts = lu * max_u_lb / max(abs(dia_u_p), abs(out_u_p))

        self.lineEdit_11.setText('{}'.format(ts))

    def open_3d_model(self):

        self.model_fileName = \
            QtWidgets.QFileDialog.getOpenFileName(self, "Open 3D ASCII STL model", os.getcwd(), "Image Files (*.stl)")[0]

        if self.model_fileName != '':
            print(self.model_fileName)

    def voxelize(self):

        resolution = int(self.spinBox.value())
        mesh, self.bounding_box = read_and_reshape_stl(self.model_fileName, resolution)

        process_number = int(self.spinBox_2.value())
        self.voxels = voxelize(mesh, self.bounding_box, process_number)

        self.pushButton_8.clicked.connect(self.generate_final_model)
        self.horizontalSlider.sliderReleased.connect(self.update_graph)
        self.label_6.setText('{}'.format(self.bounding_box[0]))
        self.label_7.setText('{}'.format(self.bounding_box[1]))
        self.label_8.setText('{}'.format(self.bounding_box[2]))
        self.horizontalSlider.setRange(0, self.bounding_box[2]-1)
        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

        self.i0 = np.array([])
        self.i1 = np.array([])
        self.i2 = np.array([])
        self.o1 = np.array([])
        self.o2 = np.array([])

    # def obtain_output_folder(self):
    #
    #     self.output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Directory",
    #                                                      os.getcwd(),
    #                                                      QtWidgets.QFileDialog.ShowDirsOnly
    #                                                      | QtWidgets.QFileDialog.DontResolveSymlinks)
    #
    #     if self.output_dir != '':
    #         print(self.output_dir)

    def generate_final_model(self):

        height = float(self.lineEdit_2.text())
        self.lb_unit = float(self.lineEdit.text())

        h = round(height / self.lb_unit)

        self.voxels = np.delete(self.voxels, np.s_[self.bounding_box[1]:], 0)

        outer = np.zeros((h, self.bounding_box[0], self.bounding_box[2]), dtype=np.int8)

        if self.comboBox.currentIndex() == 0:
            outer[:, 0, :] = 1
            outer[:, -1, :] = 1
        elif self.comboBox.currentIndex() == 1:
            outer[:, 0, :] = 3
            outer[:, -1, :] = 1
        elif self.comboBox.currentIndex() == 2:
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

                if ff == 1 and self.voxels[-1, i, j] > -1:

                    self.voxels[-1, i, j] = 1
                    self.voxels[-2, i, j] = -3

                elif self.voxels[-1, i, j] < 0:
                    ff = 0

                if bf == 1 and self.voxels[-1, i, -1 - j] > -1:
                    self.voxels[-1, i, -1 - j] = 1
                    self.voxels[-2, i, -1 - j] = -3
                elif self.voxels[-1, i, -1 - j] < 0:
                    bf = 0

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

        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def center_plot(self):

        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def rotate_x_minus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (2, 1, 0)), 0)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

        self.horizontalSlider.setRange(0, self.bounding_box[2] - 1)
        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def rotate_x_plus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 0), (2, 1, 0))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

        self.horizontalSlider.setRange(0, self.bounding_box[2] - 1)
        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def rotate_z_minus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 1), (0, 2, 1))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

        self.horizontalSlider.setRange(0, self.bounding_box[2] - 1)
        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def rotate_z_plus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (0, 2, 1)), 1)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

        self.horizontalSlider.setRange(0, self.bounding_box[2] - 1)
        self.horizontalSlider.setValue(int(np.floor(self.bounding_box[2] / 2)))
        self.update_graph()

    def update_graph(self):

        mid = int(self.horizontalSlider.value())
        plane = self.voxels[:, :, mid]

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.set_navigate(False)
        self.MplWidget.canvas.axes.imshow(plane, vmin=-2, vmax=3)
        self.MplWidget.canvas.axes.invert_yaxis()
        self.MplWidget.canvas.axes.set_xlabel('x-axis')
        self.MplWidget.canvas.axes.set_ylabel('y-axis')
        self.MplWidget.canvas.axes.set_title('cross section x/y plane at z = {}'.format(mid))
        self.MplWidget.canvas.draw()

    def start_simulation(self):

        if len(self.i0) == 0:
            [self.i0, self.i1, self.i2, self.o1, self.o2] = preprocessing.node_initialization(self.voxels
                                                                                              , self.bounding_box[1])

        self.update_graph()

        lu = float(self.lineEdit.text()) / 1000

        duration = 1
        max_u_lb = 0.05
        dia_u_p = 0.01
        out_u_p = 0.01

        frequency = 300

        interval = 2

        # duration = float(self.lineEdit_9.text()

        # max_u_lb = float(self.lineEdit_5.text())

        # dia_u_p = float(self.lineEdit_4.text())

        # out_u_p = float(self.lineEdit_7.text())

        # NuP = float(self.lineEdit_3.text())

        # frequency = float(self.lineEdit_6.text())

        # interval = float(self.lineEdit_11.text())

        ts = lu * max_u_lb / abs(dia_u_p)

        Nulb = lu * lu / ts

        dia_u_lb = dia_u_p * ts / lu

        out_u_lb = out_u_p * ts / lu

        print("Time Step: {} s".format(ts))

        tau_inv = 1

        # tau_inv = 1.0 / ((NuP / Nulb) * 3.0 + 0.5)

        print("tau_inverse: {}".format(tau_inv))

        lbm.solve(duration, lu, dia_u_lb, out_u_lb, tau_inv, ts, self.voxels, interval, frequency
                  , self.i0, self.i1, self.i2, self.o1, self.o2, 'result')


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    qt_app = MyQtApp()
    qt_app.show()
    app.exec_()