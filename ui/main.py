# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created: Sun Apr 28 20:00:40 2019
#      by: pyside2-uic 2.0.0 running on PySide2 5.6.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(983, 766)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 481, 761))
        self.tabWidget.setObjectName("tabWidget")
        self.voxelization = QtWidgets.QWidget()
        self.voxelization.setEnabled(True)
        self.voxelization.setToolTip("")
        self.voxelization.setAccessibleName("")
        self.voxelization.setObjectName("voxelization")
        self.pushButton = QtWidgets.QPushButton(self.voxelization)
        self.pushButton.setGeometry(QtCore.QRect(10, 40, 101, 21))
        self.pushButton.setObjectName("pushButton")
        self.spinBox = QtWidgets.QSpinBox(self.voxelization)
        self.spinBox.setGeometry(QtCore.QRect(320, 80, 51, 22))
        self.spinBox.setMinimum(10)
        self.spinBox.setMaximum(3500)
        self.spinBox.setObjectName("spinBox")
        self.label = QtWidgets.QLabel(self.voxelization)
        self.label.setGeometry(QtCore.QRect(10, 70, 281, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.voxelization)
        self.label_2.setGeometry(QtCore.QRect(10, 140, 61, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.voxelization)
        self.label_3.setGeometry(QtCore.QRect(10, 180, 51, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.voxelization)
        self.label_4.setGeometry(QtCore.QRect(10, 230, 21, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.voxelization)
        self.label_5.setGeometry(QtCore.QRect(10, 280, 47, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.voxelization)
        self.label_6.setGeometry(QtCore.QRect(50, 180, 71, 20))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.voxelization)
        self.label_7.setGeometry(QtCore.QRect(60, 230, 71, 21))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.voxelization)
        self.label_8.setGeometry(QtCore.QRect(50, 280, 71, 21))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.pushButton_2 = QtWidgets.QPushButton(self.voxelization)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 420, 131, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.voxelization)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 460, 131, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.voxelization)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 530, 131, 21))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.voxelization)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 570, 131, 21))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_9 = QtWidgets.QLabel(self.voxelization)
        self.label_9.setGeometry(QtCore.QRect(10, 10, 181, 20))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.voxelization)
        self.label_10.setGeometry(QtCore.QRect(10, 340, 311, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.voxelization)
        self.label_11.setGeometry(QtCore.QRect(20, 100, 231, 16))
        self.label_11.setObjectName("label_11")
        self.pushButton_6 = QtWidgets.QPushButton(self.voxelization)
        self.pushButton_6.setGeometry(QtCore.QRect(370, 120, 75, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_12 = QtWidgets.QLabel(self.voxelization)
        self.label_12.setGeometry(QtCore.QRect(20, 360, 261, 20))
        self.label_12.setObjectName("label_12")
        self.spinBox_2 = QtWidgets.QSpinBox(self.voxelization)
        self.spinBox_2.setGeometry(QtCore.QRect(280, 120, 42, 22))
        self.spinBox_2.setMinimum(1)
        self.spinBox_2.setProperty("value", 16)
        self.spinBox_2.setObjectName("spinBox_2")
        self.label_14 = QtWidgets.QLabel(self.voxelization)
        self.label_14.setGeometry(QtCore.QRect(150, 120, 81, 20))
        self.label_14.setObjectName("label_14")
        self.tabWidget.addTab(self.voxelization, "")
        self.configuration = QtWidgets.QWidget()
        self.configuration.setObjectName("configuration")
        self.gridLayout = QtWidgets.QGridLayout(self.configuration)
        self.gridLayout.setObjectName("gridLayout")
        self.label_38 = QtWidgets.QLabel(self.configuration)
        self.label_38.setObjectName("label_38")
        self.gridLayout.addWidget(self.label_38, 24, 0, 1, 1)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout.addWidget(self.lineEdit_10, 24, 3, 1, 1)
        self.pushButton_9 = QtWidgets.QPushButton(self.configuration)
        self.pushButton_9.setObjectName("pushButton_9")
        self.gridLayout.addWidget(self.pushButton_9, 25, 5, 1, 2)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.gridLayout.addWidget(self.lineEdit_12, 12, 3, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.configuration)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout.addWidget(self.comboBox_2, 23, 3, 1, 4)
        self.radioButton_3 = QtWidgets.QRadioButton(self.configuration)
        self.radioButton_3.setObjectName("radioButton_3")
        self.gridLayout.addWidget(self.radioButton_3, 17, 3, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.configuration)
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 1, 0, 1, 2)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 7, 3, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 10, 3, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.configuration)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 6, 4, 1, 2)
        self.lineEdit = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 3, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.configuration)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 0, 4, 1, 2)
        self.comboBox = QtWidgets.QComboBox(self.configuration)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 3, 3, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 6, 3, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.configuration)
        self.label_37.setObjectName("label_37")
        self.gridLayout.addWidget(self.label_37, 12, 0, 1, 1)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.gridLayout.addWidget(self.lineEdit_13, 13, 3, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.configuration)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 5, 4, 1, 2)
        self.label_20 = QtWidgets.QLabel(self.configuration)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 5, 0, 1, 2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 5, 3, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 3, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.configuration)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 6, 0, 1, 3)
        self.label_19 = QtWidgets.QLabel(self.configuration)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 1, 4, 1, 2)
        self.label_29 = QtWidgets.QLabel(self.configuration)
        self.label_29.setObjectName("label_29")
        self.gridLayout.addWidget(self.label_29, 3, 0, 1, 2)
        self.pushButton_8 = QtWidgets.QPushButton(self.configuration)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout.addWidget(self.pushButton_8, 3, 5, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.configuration)
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 11, 5, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.configuration)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 0, 0, 1, 2)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout.addWidget(self.lineEdit_9, 15, 3, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.configuration)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 9, 4, 1, 2)
        self.label_27 = QtWidgets.QLabel(self.configuration)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 9, 0, 1, 2)
        self.label_24 = QtWidgets.QLabel(self.configuration)
        self.label_24.setObjectName("label_24")
        self.gridLayout.addWidget(self.label_24, 10, 0, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.configuration)
        self.label_36.setObjectName("label_36")
        self.gridLayout.addWidget(self.label_36, 22, 5, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.configuration)
        self.label_39.setObjectName("label_39")
        self.gridLayout.addWidget(self.label_39, 13, 0, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.configuration)
        self.label_35.setObjectName("label_35")
        self.gridLayout.addWidget(self.label_35, 22, 0, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.configuration)
        self.label_31.setObjectName("label_31")
        self.gridLayout.addWidget(self.label_31, 11, 0, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.configuration)
        self.label_34.setObjectName("label_34")
        self.gridLayout.addWidget(self.label_34, 15, 5, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.configuration)
        self.label_33.setObjectName("label_33")
        self.gridLayout.addWidget(self.label_33, 23, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.configuration)
        self.label_25.setObjectName("label_25")
        self.gridLayout.addWidget(self.label_25, 7, 0, 1, 3)
        self.label_30 = QtWidgets.QLabel(self.configuration)
        self.label_30.setObjectName("label_30")
        self.gridLayout.addWidget(self.label_30, 15, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.configuration)
        self.label_26.setObjectName("label_26")
        self.gridLayout.addWidget(self.label_26, 7, 4, 1, 2)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.gridLayout.addWidget(self.lineEdit_11, 22, 3, 1, 1)
        self.pushButton_10 = QtWidgets.QPushButton(self.configuration)
        self.pushButton_10.setObjectName("pushButton_10")
        self.gridLayout.addWidget(self.pushButton_10, 25, 0, 1, 2)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout.addWidget(self.lineEdit_7, 9, 3, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.configuration)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout.addWidget(self.lineEdit_8, 11, 3, 1, 1)
        self.pushButton_11 = QtWidgets.QPushButton(self.configuration)
        self.pushButton_11.setObjectName("pushButton_11")
        self.gridLayout.addWidget(self.pushButton_11, 19, 3, 1, 1)
        self.radioButton_2 = QtWidgets.QRadioButton(self.configuration)
        self.radioButton_2.setObjectName("radioButton_2")
        self.gridLayout.addWidget(self.radioButton_2, 16, 3, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.configuration)
        self.label_32.setObjectName("label_32")
        self.gridLayout.addWidget(self.label_32, 16, 0, 1, 1)
        self.tabWidget.addTab(self.configuration, "")
        self.MplWidget = MplWidget(self.centralwidget)
        self.MplWidget.setGeometry(QtCore.QRect(490, 20, 481, 741))
        self.MplWidget.setObjectName("MplWidget")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(490, 0, 371, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(900, 0, 75, 23))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(470, 0, 21, 20))
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(870, 0, 31, 16))
        self.label_15.setObjectName("label_15")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.tabWidget, self.pushButton)
        MainWindow.setTabOrder(self.pushButton, self.spinBox)
        MainWindow.setTabOrder(self.spinBox, self.spinBox_2)
        MainWindow.setTabOrder(self.spinBox_2, self.pushButton_6)
        MainWindow.setTabOrder(self.pushButton_6, self.pushButton_2)
        MainWindow.setTabOrder(self.pushButton_2, self.pushButton_3)
        MainWindow.setTabOrder(self.pushButton_3, self.pushButton_4)
        MainWindow.setTabOrder(self.pushButton_4, self.pushButton_5)
        MainWindow.setTabOrder(self.pushButton_5, self.horizontalSlider)
        MainWindow.setTabOrder(self.horizontalSlider, self.pushButton_7)
        MainWindow.setTabOrder(self.pushButton_7, self.lineEdit)
        MainWindow.setTabOrder(self.lineEdit, self.lineEdit_2)
        MainWindow.setTabOrder(self.lineEdit_2, self.comboBox)
        MainWindow.setTabOrder(self.comboBox, self.pushButton_8)
        MainWindow.setTabOrder(self.pushButton_8, self.lineEdit_3)
        MainWindow.setTabOrder(self.lineEdit_3, self.lineEdit_4)
        MainWindow.setTabOrder(self.lineEdit_4, self.lineEdit_6)
        MainWindow.setTabOrder(self.lineEdit_6, self.lineEdit_7)
        MainWindow.setTabOrder(self.lineEdit_7, self.lineEdit_8)
        MainWindow.setTabOrder(self.lineEdit_8, self.comboBox_2)
        MainWindow.setTabOrder(self.comboBox_2, self.pushButton_10)
        MainWindow.setTabOrder(self.pushButton_10, self.pushButton_9)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "3D Nozzle Lattice Tool", None, -1))
        self.pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "open 3D model", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWindow", "2.Input the resolution of the longest dimension of model:", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWindow", "Resolution:", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("MainWindow", "x-", None, -1))
        self.label_4.setText(QtWidgets.QApplication.translate("MainWindow", "y-", None, -1))
        self.label_5.setText(QtWidgets.QApplication.translate("MainWindow", "z-", None, -1))
        self.pushButton_2.setText(QtWidgets.QApplication.translate("MainWindow", "x axis  + 90°  rotation", None, -1))
        self.pushButton_3.setText(QtWidgets.QApplication.translate("MainWindow", "x axis  - 90°  rotation", None, -1))
        self.pushButton_4.setText(QtWidgets.QApplication.translate("MainWindow", "z axis  + 90°  rotation", None, -1))
        self.pushButton_5.setText(QtWidgets.QApplication.translate("MainWindow", "z axis  + 90°  rotation", None, -1))
        self.label_9.setText(QtWidgets.QApplication.translate("MainWindow", "1. Open 3D ASCII STL model", None, -1))
        self.label_10.setText(QtWidgets.QApplication.translate("MainWindow", "3. Rotate model so the nozzel exit is point in the + y direction", None, -1))
        self.label_11.setText(QtWidgets.QApplication.translate("MainWindow", "Then Click on \" Voxelize \"  Button", None, -1))
        self.pushButton_6.setText(QtWidgets.QApplication.translate("MainWindow", "Voxelize", None, -1))
        self.label_12.setText(QtWidgets.QApplication.translate("MainWindow", "horizontal flow in the x direction", None, -1))
        self.label_14.setText(QtWidgets.QApplication.translate("MainWindow", "Process Number", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.voxelization), QtWidgets.QApplication.translate("MainWindow", "voxelization", None, -1))
        self.label_38.setText(QtWidgets.QApplication.translate("MainWindow", "TextLabel", None, -1))
        self.pushButton_9.setText(QtWidgets.QApplication.translate("MainWindow", "Start Simulation", None, -1))
        self.comboBox_2.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "Plot", None, -1))
        self.comboBox_2.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "Date File", None, -1))
        self.comboBox_2.setItemText(2, QtWidgets.QApplication.translate("MainWindow", "Both", None, -1))
        self.radioButton_3.setText(QtWidgets.QApplication.translate("MainWindow", "LB Pressure", None, -1))
        self.label_18.setText(QtWidgets.QApplication.translate("MainWindow", "outter region height", None, -1))
        self.label_23.setText(QtWidgets.QApplication.translate("MainWindow", "mm / s", None, -1))
        self.label_17.setText(QtWidgets.QApplication.translate("MainWindow", "mm", None, -1))
        self.comboBox.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "No Outter Flow", None, -1))
        self.comboBox.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "Left to Right", None, -1))
        self.comboBox.setItemText(2, QtWidgets.QApplication.translate("MainWindow", "Right to Left", None, -1))
        self.label_37.setText(QtWidgets.QApplication.translate("MainWindow", "Output Plane: y", None, -1))
        self.label_21.setText(QtWidgets.QApplication.translate("MainWindow", "mm2 / s", None, -1))
        self.label_20.setText(QtWidgets.QApplication.translate("MainWindow", "kinematic viscosity", None, -1))
        self.label_22.setText(QtWidgets.QApplication.translate("MainWindow", "diaphragm max velocity", None, -1))
        self.label_19.setText(QtWidgets.QApplication.translate("MainWindow", "mm", None, -1))
        self.label_29.setText(QtWidgets.QApplication.translate("MainWindow", "outter flow option", None, -1))
        self.pushButton_8.setText(QtWidgets.QApplication.translate("MainWindow", "Update Model", None, -1))
        self.label_40.setText(QtWidgets.QApplication.translate("MainWindow", "Ex. 1, 2, 3", None, -1))
        self.label_16.setText(QtWidgets.QApplication.translate("MainWindow", "lacttice unit lenth", None, -1))
        self.label_28.setText(QtWidgets.QApplication.translate("MainWindow", "mm / s", None, -1))
        self.label_27.setText(QtWidgets.QApplication.translate("MainWindow", "outter flow velocity", None, -1))
        self.label_24.setText(QtWidgets.QApplication.translate("MainWindow", "max velocity in LB Unit", None, -1))
        self.label_36.setText(QtWidgets.QApplication.translate("MainWindow", "s", None, -1))
        self.label_39.setText(QtWidgets.QApplication.translate("MainWindow", "Output Plane: z", None, -1))
        self.label_35.setText(QtWidgets.QApplication.translate("MainWindow", "Output Interval", None, -1))
        self.label_31.setText(QtWidgets.QApplication.translate("MainWindow", "Output Plane: x", None, -1))
        self.label_34.setText(QtWidgets.QApplication.translate("MainWindow", "s", None, -1))
        self.label_33.setText(QtWidgets.QApplication.translate("MainWindow", "Output Type", None, -1))
        self.label_25.setText(QtWidgets.QApplication.translate("MainWindow", "diaphragm frequency", None, -1))
        self.label_30.setText(QtWidgets.QApplication.translate("MainWindow", "Simulation Duration", None, -1))
        self.label_26.setText(QtWidgets.QApplication.translate("MainWindow", "Hz", None, -1))
        self.pushButton_10.setText(QtWidgets.QApplication.translate("MainWindow", "End Process", None, -1))
        self.pushButton_11.setText(QtWidgets.QApplication.translate("MainWindow", "show time step", None, -1))
        self.radioButton_2.setText(QtWidgets.QApplication.translate("MainWindow", "Velocity", None, -1))
        self.label_32.setText(QtWidgets.QApplication.translate("MainWindow", "Output Properties", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.configuration), QtWidgets.QApplication.translate("MainWindow", "configuration", None, -1))
        self.pushButton_7.setText(QtWidgets.QApplication.translate("MainWindow", "center", None, -1))
        self.label_13.setText(QtWidgets.QApplication.translate("MainWindow", "z   0", None, -1))
        self.label_15.setText(QtWidgets.QApplication.translate("MainWindow", "max", None, -1))

from ui.mplwidget import MplWidget
