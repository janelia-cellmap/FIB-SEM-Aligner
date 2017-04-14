# to run this program
# in the command window
# >jupyter qtconsole
#: run Build_Stack.py


import os
import numpy as np
import glob
from skimage.external import tifffile
from skimage import io
from skimage.transform import rescale

import sys
import time

import gc


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


print('Finished loading modules')

#This is what %pylab does:
# import numpy
# import matplotlib
# from matplotlib import pylab, mlab, pyplot
# np = numpy
# plt = pyplot
# from IPython.core.pylabtools import figsize, getfigs
# from pylab import *
# from numpy import *

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


class FileList_Window(QWidget):
    def __init__(self):
        print('started initializing window')
        QWidget.__init__(self)
        layout = QGridLayout(self)
        self.frame_names = []
        self.file_mask_label = QLabel("File Name Filter:")
        layout.addWidget(self.file_mask_label, 0, 0)

        self.file_mask = QComboBox()
        layout.addWidget(self.file_mask, 0, 1, 1, 4)
        self.file_mask.setEditable(True)
        self.file_mask.addItem("0/0.png")

        self.button_browse = QPushButton('Select Directory', self)
        layout.addWidget(self.button_browse, 2, 1)
        self.button_browse.clicked.connect(self.handleButton_browse)

        self.dir_label = QLabel("Search Directory:")
        layout.addWidget(self.dir_label, 1, 0)

        self.search_dir = QComboBox()
        layout.addWidget(self.search_dir, 1, 1, 1, 4)
        self.search_dir.setEditable(True)
        self.search_dir.addItem(QDir.currentPath())

        self.button_find = QPushButton('Re-Find Files', self)
        layout.addWidget(self.button_find, 2, 4)
        self.button_find.clicked.connect(self.handleButton_find)

        self.File_Table = QTableWidget()
        layout.addWidget(self.File_Table, 3, 0, 5, 5)
        self.File_Table.setColumnCount(1)
        self.File_Table.setHorizontalHeaderLabels(['Selected Files'])
        self.File_Table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        self.button_remove_row = QPushButton('Remove Selected File from the List', self)
        layout.addWidget(self.button_remove_row, 8, 1)
        self.button_remove_row.clicked.connect(self.handleButton_remove)

        self.button_add_row = QPushButton('Add a File', self)
        layout.addWidget(self.button_add_row, 8, 2)
        self.button_add_row.clicked.connect(self.handleButton_add)

        self.button_transform = QPushButton('Transform Files', self)
        layout.addWidget(self.button_transform, 8, 4)
        self.button_transform.clicked.connect(self.handleButton_transform)

        self.target_file_label = QLabel("Destination File (.tif):")
        layout.addWidget(self.target_file_label, 9, 0)

        self.target_filename = QComboBox()
        layout.addWidget(self.target_filename, 9, 1, 1, 4)
        self.target_filename.setEditable(True)
        self.target_filename.addItem("transfer_target.tif")

        self.XY_bin_label = QLabel("X-Y binninng")
        layout.addWidget(self.XY_bin_label, 10, 0)
        self.XY_bin_val = QLineEdit('1')
        layout.addWidget(self.XY_bin_val, 10, 1, 1, 1)

        self.Z_bin_label = QLabel("Z binninng")
        layout.addWidget(self.Z_bin_label, 10, 3)
        self.Z_bin_val = QLineEdit('4')
        layout.addWidget(self.Z_bin_val, 10, 4, 1, 1)
            
        self.Xi_lab = QLabel("Left")
        layout.addWidget(self.Xi_lab, 11, 0)
        self.Xi_val = QLineEdit('0')
        layout.addWidget(self.Xi_val, 11, 1, 1, 1)
  
        self.Xa_lab = QLabel("Right")
        layout.addWidget(self.Xa_lab, 11, 3)
        self.Xa_val = QLineEdit('2500')
        layout.addWidget(self.Xa_val, 11, 4, 1, 1)

        self.Yi_lab = QLabel("Top")
        layout.addWidget(self.Yi_lab, 12, 0)
        self.Yi_val = QLineEdit('0')
        layout.addWidget(self.Yi_val, 12, 1, 1, 1)

        self.Ya_lab = QLabel("Bottom")
        layout.addWidget(self.Ya_lab, 12, 3)
        self.Ya_val = QLineEdit('2500')
        layout.addWidget(self.Ya_val, 12, 4, 1, 1)

        self.button_test_crop = QPushButton('Test Crop', self)
        layout.addWidget(self.button_test_crop, 13, 2)
        self.button_test_crop.clicked.connect(self.handleButton_test_crop)


       # Create progressBar.
        self.progr_bar = QProgressBar()
        layout.addWidget(self.progr_bar, 14, 0, 1, 5)
        self.progr_bar.setValue(0)
        
        # self.img_lab = QLabel()
        # layout.addWidget(self.img_lab, 14, 0)
        # print(os.getcwd() + '1.png')
        # print(self.img_lab.size())
        # Pixmap = QPixmap(os.getcwd() + '/0.png')
        # myScaledPixmap = Pixmap.scaled(self.img_lab.size(), Qt.KeepAspectRatio)
        # self.img_lab.setPixmap(Pixmap)
        print('finished initializing window')


    def handleButton_browse(self):
        title = 'Select Directory'
        search_directory = QFileDialog.getExistingDirectory(self, title)
        print('Setting Search Directory to' + search_directory)
        if not search_directory == '':
            self.search_dir.addItem(search_directory)
            self.search_dir.setCurrentIndex(self.search_dir.count() - 1)
            # os.chdir(search_directory)

    def fill_file_table(self):
        n_files = len(self.frame_names)
        if n_files > 0:
            indices = np.arange(n_files)
            self.File_Table.setRowCount(n_files)
            for chunk_name, ind in zip(self.frame_names, indices):
                self.File_Table.setItem(ind, 0, QTableWidgetItem(chunk_name))
            print('Table Populated')

    def handleButton_remove(self):
        current_row = self.File_Table.currentRow()
        print(current_row)
        del self.frame_names[current_row]
        self.fill_file_table()

    def handleButton_add(self):
        filename_to_add =  QFileDialog.getOpenFileName(self, 'Select a File to add')
        self.frame_names.append(filename_to_add)
        self.fill_file_table()

    def handleButton_find(self):
        search_directory = self.search_dir.currentText()
        file_mask = self.file_mask.currentText()
        print("Search directory:     " + search_directory)
        print("File mask:            " + file_mask)
        # os.chdir(search_directory)
        self.frame_names = glob.glob(search_directory + '/**/*' + file_mask,recursive=True)
        # self.frame_names = sorted(self.frame_names)
        fr_num = [int(fr_name.split("\\")[-3]) for fr_name in self.frame_names ]
        print("Found ",len(self.frame_names)," files, sorting")
        self.frame_names = np.asarray([xx for (yy,xx) in sorted(zip(fr_num,self.frame_names))])
        self.fill_file_table()

    def handleButton_test_crop(self):
        XY_bin = int(self.XY_bin_val.text())
        Z_bin = int(self.Z_bin_val.text())

        xi = int(self.Xi_val.text())
        xa = int(self.Xa_val.text())
        yi = int(self.Yi_val.text())
        ya = int(self.Ya_val.text())
        dx = xa - xi
        dy = ya - yi

        crop = (slice(yi, ya), slice(xi, xa))
        nsh = (1,dy/XY_bin,dx/XY_bin)
        print(nsh)

        n_files = np.floor_divide(len(self.frame_names), Z_bin) * Z_bin
        dz = n_files/Z_bin

        if n_files > 0:
            indices = np.arange(dz)
            frame_names_2D = self.frame_names[0:n_files].reshape(-1,Z_bin)
            target_file_name = self.target_filename.currentText()
            test_file_name = target_file_name.replace('.tif','_test.tif')
            target_file_UX, test_name = os.path.split(test_file_name)
            print('Target Directory', target_file_UX)
            if target_file_UX == '':
                target_file_UX = QDir.currentPath()
            else:
                try:
                    os.makedirs(target_file_UX, exist_ok=True)
                except :
                    pass
                # you should use a context manager here
                # with open(target_file_name, 'wb+') as f_data:
                #   stuff block...
            print('Starting Transfer into: ', test_file_name)


            
            with tifffile.TiffWriter(test_file_name, bigtiff=True) as tif:
                chunk_names = frame_names_2D[0,:]
                print('Transferring data files:  ', chunk_names)
                x = np.array([io.imread(fr)[yi:ya,xi:xa] for fr in chunk_names])
                # new_frame = new_frame/Z_bin
                frame_subset = bin_ndarray(np.asarray(x,dtype=np.float32), new_shape=nsh, operation='mean')
                tif.save(frame_subset.astype(x.dtype))
                chunk_names = frame_names_2D[-1,:]
                print('Transferring data files:  ', chunk_names)
                x = np.array([io.imread(fr)[yi:ya,xi:xa] for fr in chunk_names])
                # new_frame = new_frame/Z_bin
                frame_subset = bin_ndarray(np.asarray(x,dtype=np.float32), new_shape=nsh, operation='mean')
                tif.save(frame_subset.astype(x.dtype))
                tif.close()
        return

    def handleButton_transform(self):
        XY_bin = int(self.XY_bin_val.text())
        Z_bin = int(self.Z_bin_val.text())
        print("Starting Transformations, binning set at:",XY_bin, Z_bin)
        n_files = np.floor_divide(len(self.frame_names), Z_bin) * Z_bin
        print("Will use ",n_files," files")

        xi = int(self.Xi_val.text())
        xa = int(self.Xa_val.text())
        yi = int(self.Yi_val.text())
        ya = int(self.Ya_val.text())
        dx = xa - xi
        dy = ya - yi
        dz = n_files/Z_bin

        crop = (slice(yi, ya), slice(xi, xa))
        
        if n_files > 0:
            indices = np.arange(dz)
            frame_names_2D = self.frame_names[0:n_files].reshape(-1,Z_bin)
            target_file_name = self.target_filename.currentText()
            target_file_UX, target_name = os.path.split(target_file_name)
            print('Target Directory', target_file_UX)
            if target_file_UX == '':
                target_file_UX = QDir.currentPath()
            else:
                try:
                    os.makedirs(target_file_UX, exist_ok=True)
                except :
                    pass
            # you should use a context manager here
            # with open(target_file_name, 'wb+') as f_data:
            #   stuff block...
            print('Starting Transfer')

            nframes = 0
            progr_bar_val_prev = 0
            self.progr_bar.setValue(progr_bar_val_prev)
            nsh = (1,dy/XY_bin,dx/XY_bin)
            print('New shape: ',nsh)
            with tifffile.TiffWriter(target_file_name, bigtiff=True) as tif:
                for chunk_names, cnt in zip(frame_names_2D, indices):
                    # frame_subset = np.zeros((Z_bin, dy,dx,), dtype=float)
                    # for fr in chunk_name:
                        # new_frame+= io.imread(fr)[yi:ya,xi:xa]
                    print('Transferring data files:  ', chunk_names)
                    x = np.array([io.imread(fr)[yi:ya,xi:xa] for fr in chunk_names])
                    # new_frame = new_frame/Z_bin
                    frame_subset = bin_ndarray(np.asarray(x,dtype=np.float32), new_shape=nsh, operation='mean')
                    tif.save(frame_subset.astype(x.dtype))
                    progr_bar_val = (cnt + 1) / dz * 100
                    if progr_bar_val - progr_bar_val_prev >= 1:
                        self.progr_bar.setValue(progr_bar_val)
                        QApplication.processEvents()
                        progr_bar_val_prev = progr_bar_val
                tif.close()
            self.progr_bar.setValue(100)
        return

def main():

    app = QApplication(sys.argv)
    frame_names = []
    window = FileList_Window()

    window.resize(800, 800)
    window.setWindowTitle('Select TIFF Files and press Transform')
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    print('Starting GUI')
    main()
