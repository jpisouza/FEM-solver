from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
from Main_GUI import Main
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pyvista as pv
from pyvistaqt import QtInteractor

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal()
    sim_progress = pyqtSignal(int)

    def set_simulation(self) :
        Main.set_simulation(self.case,self.table,self.textEdit_Re,self.textEdit_Pr,self.textEdit_Ga,self.textEdit_Gr,self.textEdit_dt,self.textEdit_dt_2,self.convection)
        if self.particles:
            Main.particles = True
            Main.set_simulation_particles(self.case,self.particles_D,self.particles_rho,self.particles_nLoop,self.particles_nparticles,self.particles_lim_inf_x,self.particles_lim_sup_x,self.particles_lim_inf_y,self.particles_lim_sup_y, self.textEdit_dt_2)
        else:
            Main.particles = False
    def runPlay(self):
        self.progress.emit()
        self.set_simulation()
        while(True):
            if self.status == 'stop':
                break
            Main.play(self.iteration) 
            self.sim_progress.emit(self.iteration)
            self.iteration+=1            
        self.finished.emit()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(804, 748)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setDocumentMode(True)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 802, 715))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setUsesScrollButtons(True)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.Simtab = QtWidgets.QWidget()
        self.Simtab.setObjectName("Simtab")
        self.tableWidget = QtWidgets.QTableWidget(self.Simtab)
        self.tableWidget.setGeometry(QtCore.QRect(80, 310, 631, 192))
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.textEdit = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit.setGeometry(QtCore.QRect(230, 100, 341, 31))
        self.textEdit.setObjectName("textEdit")
        self.toolButton = QtWidgets.QToolButton(self.Simtab)
        self.toolButton.setGeometry(QtCore.QRect(170, 100, 51, 31))
        self.toolButton.setObjectName("toolButton")
        self.label_2 = QtWidgets.QLabel(self.Simtab)
        self.label_2.setGeometry(QtCore.QRect(330, 70, 131, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textEdit_Re = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Re.setGeometry(QtCore.QRect(130, 260, 71, 31))
        self.textEdit_Re.setObjectName("textEdit_Re")
        self.textEdit_Pr = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Pr.setGeometry(QtCore.QRect(290, 260, 71, 31))
        self.textEdit_Pr.setObjectName("textEdit_Pr")
        self.textEdit_Ga = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Ga.setGeometry(QtCore.QRect(450, 260, 71, 31))
        self.textEdit_Ga.setObjectName("textEdit_Ga")
        self.textEdit_Gr = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Gr.setGeometry(QtCore.QRect(610, 260, 71, 31))
        self.textEdit_Gr.setObjectName("textEdit_Gr")
        self.label_3 = QtWidgets.QLabel(self.Simtab)
        self.label_3.setGeometry(QtCore.QRect(200, 170, 381, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.Simtab)
        self.label_4.setGeometry(QtCore.QRect(100, 270, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.Simtab)
        self.label_5.setGeometry(QtCore.QRect(260, 270, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_Ga = QtWidgets.QLabel(self.Simtab)
        self.label_Ga.setGeometry(QtCore.QRect(420, 270, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga.setFont(font)
        self.label_Ga.setObjectName("label_Ga")
        self.label_Ga_2 = QtWidgets.QLabel(self.Simtab)
        self.label_Ga_2.setGeometry(QtCore.QRect(580, 270, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga_2.setFont(font)
        self.label_Ga_2.setObjectName("label_Ga_2")
        self.pushButton = QtWidgets.QPushButton(self.Simtab)
        self.pushButton.setGeometry(QtCore.QRect(350, 20, 93, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_6 = QtWidgets.QLabel(self.Simtab)
        self.label_6.setGeometry(QtCore.QRect(310, 530, 131, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.textEdit_D = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_D.setGeometry(QtCore.QRect(70, 590, 71, 31))
        self.textEdit_D.setObjectName("textEdit_D")
        self.label_D = QtWidgets.QLabel(self.Simtab)
        self.label_D.setGeometry(QtCore.QRect(40, 600, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_D.setFont(font)
        self.label_D.setObjectName("label_D")
        self.label_Ga_3 = QtWidgets.QLabel(self.Simtab)
        self.label_Ga_3.setGeometry(QtCore.QRect(300, 600, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga_3.setFont(font)
        self.label_Ga_3.setObjectName("label_Ga_3")
        self.label_8 = QtWidgets.QLabel(self.Simtab)
        self.label_8.setGeometry(QtCore.QRect(160, 600, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.textEdit_Ga_2 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Ga_2.setGeometry(QtCore.QRect(360, 590, 71, 31))
        self.textEdit_Ga_2.setObjectName("textEdit_Ga_2")
        self.label_Ga_4 = QtWidgets.QLabel(self.Simtab)
        self.label_Ga_4.setGeometry(QtCore.QRect(460, 600, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga_4.setFont(font)
        self.label_Ga_4.setObjectName("label_Ga_4")
        self.textEdit_rho = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_rho.setGeometry(QtCore.QRect(200, 590, 71, 31))
        self.textEdit_rho.setObjectName("textEdit_rho")
        self.textEdit_Gr_2 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Gr_2.setGeometry(QtCore.QRect(520, 590, 71, 31))
        self.textEdit_Gr_2.setObjectName("textEdit_Gr_2")
        self.textEdit_Ga_3 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Ga_3.setGeometry(QtCore.QRect(360, 630, 71, 31))
        self.textEdit_Ga_3.setObjectName("textEdit_Ga_3")
        self.label_Ga_5 = QtWidgets.QLabel(self.Simtab)
        self.label_Ga_5.setGeometry(QtCore.QRect(460, 640, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga_5.setFont(font)
        self.label_Ga_5.setObjectName("label_Ga_5")
        self.label_Ga_6 = QtWidgets.QLabel(self.Simtab)
        self.label_Ga_6.setGeometry(QtCore.QRect(300, 640, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_Ga_6.setFont(font)
        self.label_Ga_6.setObjectName("label_Ga_6")
        self.textEdit_Gr_3 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Gr_3.setGeometry(QtCore.QRect(520, 630, 71, 31))
        self.textEdit_Gr_3.setObjectName("textEdit_Gr_3")
        self.textEdit_Re_3 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Re_3.setGeometry(QtCore.QRect(680, 590, 71, 31))
        self.textEdit_Re_3.setObjectName("textEdit_Re_3")
        self.label_D_2 = QtWidgets.QLabel(self.Simtab)
        self.label_D_2.setGeometry(QtCore.QRect(610, 600, 81, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_D_2.setFont(font)
        self.label_D_2.setObjectName("label_D_2")
        self.label_D_3 = QtWidgets.QLabel(self.Simtab)
        self.label_D_3.setGeometry(QtCore.QRect(620, 640, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_D_3.setFont(font)
        self.label_D_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_D_3.setObjectName("label_D_3")
        self.textEdit_Re_4 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_Re_4.setGeometry(QtCore.QRect(680, 630, 71, 31))
        self.textEdit_Re_4.setObjectName("textEdit_Re_4")
        self.label_7 = QtWidgets.QLabel(self.Simtab)
        self.label_7.setGeometry(QtCore.QRect(630, 65, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.textEdit_dt = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_dt.setGeometry(QtCore.QRect(650, 100, 71, 31))
        self.textEdit_dt.setObjectName("textEdit_dt")
        self.checkBox_conv = QtWidgets.QCheckBox(self.Simtab)
        self.checkBox_conv.setGeometry(QtCore.QRect(350, 220, 111, 20))
        self.checkBox_conv.setObjectName("checkBox_conv")
        self.checkBox_particles = QtWidgets.QCheckBox(self.Simtab)
        self.checkBox_particles.setGeometry(QtCore.QRect(160, 530, 111, 20))
        self.checkBox_particles.setObjectName("checkBox_particles")
        self.label_9 = QtWidgets.QLabel(self.Simtab)
        self.label_9.setGeometry(QtCore.QRect(620, 140, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.textEdit_dt_2 = QtWidgets.QTextEdit(self.Simtab)
        self.textEdit_dt_2.setGeometry(QtCore.QRect(650, 170, 71, 31))
        self.textEdit_dt_2.setObjectName("textEdit_dt_2")
        self.tabWidget.addTab(self.Simtab, "")
        self.Outputtab = QtWidgets.QWidget()
        self.Outputtab.setObjectName("Outputtab")
        self.tabWidget.addTab(self.Outputtab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 804, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_simulation = QtWidgets.QAction(MainWindow)
        self.actionOpen_simulation.setObjectName("actionOpen_simulation")
        self.actionSave_simulation_settings = QtWidgets.QAction(MainWindow)
        self.actionSave_simulation_settings.setObjectName("actionSave_simulation_settings")
        self.menuFile.addAction(self.actionOpen_simulation)
        self.menuFile.addAction(self.actionSave_simulation_settings)
        self.menubar.addAction(self.menuFile.menuAction())
        
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout(self.Outputtab)
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.textEdit.setReadOnly(True)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.Outputtab)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 0, 93, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setEnabled(False)
        
        self.pushButton_3 = QtWidgets.QPushButton(self.Outputtab)
        self.pushButton_3.setGeometry(QtCore.QRect(430, 0, 93, 31))
        self.pushButton_3.setObjectName("pushButton_2")
        self.pushButton_3.setEnabled(False)
        
        self.comboBox = QtWidgets.QComboBox(self.Outputtab)
        self.comboBox.setGeometry(QtCore.QRect(570, 0, 101, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setEnabled(False)
        
        self.init_flag = True
        
        self.toolButton.clicked.connect(self.getfile)
        self.pushButton.clicked.connect(self.play)
        self.pushButton_2.clicked.connect(lambda: self.plotter.view_xy())
        self.actionOpen_simulation.triggered.connect(self.loadSimulation)
        self.actionSave_simulation_settings.triggered.connect(self.saveSimulation)
        self.checkBox_conv.stateChanged.connect(self.conv_or_exp)
        self.checkBox_particles.stateChanged.connect(self.set_particles)
        
        
        self.tableWidget.setVisible(False)
        self.checkBox_particles.setChecked(True)
        self.simulation_status = 'stop'
        
    
        
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def adjust(self):
        if str(self.comboBox.currentText()) == 'v_x':
            self.plotter.update_scalar_bar_range([np.min(Main.fluid.vx[:self.size]),np.max(Main.fluid.vx[:self.size])], name = 'Field')
        elif str(self.comboBox.currentText()) == 'v_y':
            self.plotter.update_scalar_bar_range([np.min(Main.fluid.vy[:self.size]),np.max(Main.fluid.vy[:self.size])], name = 'Field')
        elif str(self.comboBox.currentText()) == 'p':
            self.plotter.update_scalar_bar_range([np.min(self.p[:self.size]),np.max(Main.fluid.p[:self.size])], name = 'Field')
        else:
            self.plotter.update_scalar_bar_range([np.min(self.T[:self.size]),np.max(Main.fluid.T[:self.size])], name = 'Field')
            
    def change_field(self):
        if str(self.comboBox.currentText()) == 'v_x':
            self.plotter.update_scalars(Main.fluid.vx[:self.size], mesh=self.grid)
        elif str(self.comboBox.currentText()) == 'v_y':
            self.plotter.update_scalars(Main.fluid.vy[:self.size], mesh=self.grid)
        elif str(self.comboBox.currentText()) == 'p':
            self.plotter.update_scalars(self.p[:self.size], mesh=self.grid)
        else:
            self.plotter.update_scalars(self.T[:self.size], mesh=self.grid)
            
        self.adjust()
            
    def display_grid(self, i):
        file = os.path.dirname(os.path.abspath(self.fname)) + '/Results/sol-' + str(i) + '.vtk'
        if Main.MESH.mesh_kind == 'quad':  
            self.p = Main.fluid.p_quad
            self.T = Main.fluid.T_quad
        else:
            self.p = Main.fluid.p
            self.T = Main.fluid.T 
        if self.init_flag and self.simulation_status == 'play': 
            if Main.MESH.mesh_kind == 'quad':  
                self.size = Main.MESH.npoints
            else:
                self.size = Main.MESH.npoints_p 
            self.plotter.clear()
            self.grid = pv.read(file)
            self.plotter.show_axes()
            self.plotter.add_mesh(self.grid, scalars = Main.fluid.vx[:self.size], show_edges=False,cmap='jet')
            self.plotter.add_scalar_bar(title = 'Field', title_font_size = 16, n_labels = 5,label_font_size=16)
            if self.particles:
                file_part = os.path.dirname(os.path.abspath(self.fname)) + '/Results/sol_particles' + str(i) + '.vtu'
                self.part = pv.read(file_part)
                self.plotter.add_mesh(self.part, color = '#909090', style='points', point_size = 10.0, render_points_as_spheres=True, render = True)
            self.pushButton_2.setEnabled(True)
            self.comboBox.setEnabled(True)
            self.pushButton_3.setEnabled(True)
            self.init_flag = False
        else:
            if self.particles:
                self.plotter.update_coordinates(np.block([Main.particleCloud.x,0.01*np.ones((Main.particleCloud.x.shape[0],1), dtype='float')]), mesh=self.part)
            if str(self.comboBox.currentText()) == 'v_x':
                self.plotter.update_scalars(Main.fluid.vx[:self.size], mesh=self.grid)
            elif str(self.comboBox.currentText()) == 'v_y':
                self.plotter.update_scalars(Main.fluid.vy[:self.size], mesh=self.grid)
            elif str(self.comboBox.currentText()) == 'p':
                self.plotter.update_scalars(self.p[:self.size], mesh=self.grid)
            else:
                self.plotter.update_scalars(self.T[:self.size], mesh=self.grid)
                
    def play(self):
        
        if self.simulation_status == 'stop':
            
            self.simulation_status = 'play'
            
            self.toolButton.setEnabled(False)
            self.pushButton.setText('Stop')
            
            self.label_3.setText("Simulation started!")
             # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = Worker()
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            
            self.set_parameters(self.worker)
            self.worker.status = 'play'
            
            self.thread.started.connect(self.worker.runPlay)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(lambda: print('--------Simulation started!--------\n'))
            self.worker.progress.connect(lambda: self.label_3.setText('Simulation started!'))
            self.worker.sim_progress.connect(lambda i: self.label_3.setText('Iteration: ' + str(i)))
            self.worker.sim_progress.connect(self.display_grid)
            self.worker.finished.connect(lambda: self.label_3.setText("Simulation finished!"))
            self.worker.finished.connect(lambda: self.pushButton.setEnabled(True))
            self.worker.finished.connect(lambda: self.toolButton.setEnabled(True))
            self.worker.finished.connect(lambda: print('--------Simulation finished!--------\n'))
            self.comboBox.currentTextChanged.connect(self.change_field)
            self.pushButton_3.clicked.connect(self.adjust)
            # Step 6: Start the thread
            self.thread.start()
            
            
        else:
            self.worker.status = 'stop'
            self.simulation_status = 'stop'
            self.toolButton.setEnabled(True)
            self.pushButton.setText('Run')
            self.worker.deleteLater
            self.thread.deleteLater
            
            self.init_flag = True
            
            # self.pushButton_2.setEnabled(False)
            # self.comboBox.setEnabled(False)
            # self.pushButton_3.setEnabled(False)
    
    def set_parameters(self,worker):
        worker.case = self.fname
        worker.table = self.tableWidget
        worker.textEdit_Re = self.textEdit_Re
        worker.textEdit_Pr = self.textEdit_Pr
        worker.textEdit_Ga = self.textEdit_Ga
        if self.checkBox_conv.isChecked():
            worker.convection = True
        else:
            worker.convection = False
        worker.textEdit_Gr = self.textEdit_Gr
        worker.textEdit_dt = self.textEdit_dt
        worker.textEdit_dt_2 = self.textEdit_dt_2
        if self.textEdit_dt_2.toPlainText() != '':
            worker.iteration = int(self.textEdit_dt_2.toPlainText()) + 1
        else:
            worker.iteration = 0
        
        if self.particles:
           worker.particles = True
           worker.particles_D = float(self.textEdit_D.toPlainText())
           worker.particles_rho = float(self.textEdit_rho.toPlainText())
           worker.particles_nLoop = int(self.textEdit_Re_4.toPlainText())
           worker.particles_nparticles = int(self.textEdit_Re_3.toPlainText())
           worker.particles_lim_inf_x = float(self.textEdit_Ga_2.toPlainText())
           worker.particles_lim_sup_x = float(self.textEdit_Ga_3.toPlainText())
           worker.particles_lim_inf_y = float(self.textEdit_Gr_2.toPlainText())
           worker.particles_lim_sup_y = float(self.textEdit_Gr_3.toPlainText())
        else:
           worker.particles = False
        
    def getfile(self):
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
            
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open file', 'C:','(*.msh)')
        Main.def_MESH(self.fname)
        
        for i in range(len(Main.boundNames)-1):
            rowPosition = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowPosition)
            item = QtWidgets.QTableWidgetItem(Main.boundNames[i])
            self.tableWidget.setVerticalHeaderItem(i, item)
  
            
        self.tableWidget.setVisible(True)
        self.textEdit.setText(self.fname)
        self.textEdit.setReadOnly(True)

        for i in range(self.tableWidget.rowCount()):
            checkbox = QtWidgets.QCheckBox()
            self.tableWidget.setCellWidget(i, 5, checkbox)

        
        self.label_3.setText("Case loaded!")
    
    def saveSimulation(self):
        save_name,_ = QtWidgets.QFileDialog.getSaveFileName(MainWindow, 'Save file', 'C:','(*.xml)')
        data = ET.Element('Case')

        if os.path.dirname(save_name).replace('\\', '/') == os.path.dirname(os.path.abspath(self.fname)).replace('\\', '/'):
            data.set('name', os.path.basename(os.path.abspath(self.fname)).split('.')[0])
            
        else:
            data.set('name',os.path.abspath(self.fname).replace('\\', '/'))

        parameters = ET.SubElement(data,'Parameters')

        Ga = str(self.textEdit_Ga.toPlainText())
        Re = str(self.textEdit_Re.toPlainText())
        Pr = str(self.textEdit_Pr.toPlainText())
        Gr = str(self.textEdit_Gr.toPlainText())

        if self.checkBox_conv.isChecked():
            Fr_flag = 'False'
            parameters.set('Ga',Ga)
        else:
            Fr_flag = 'True'
            parameters.set('Fr',Ga)
        
        if self.checkBox_particles.isChecked():
            particles = 'True'
        else:
            particles = 'False'
        
        parameters.set('Re',Re)
        parameters.set('Gr',Gr)
        parameters.set('Pr',Pr)
        parameters.set('Fr_flag',Fr_flag)
        
        parameters.set('particles',particles)

        boundaryCond = ET.SubElement(data,'BoundaryCondition')

        OF_flag = False
        for i in range(self.tableWidget.rowCount()):
            j = int(self.tableWidget.item(i,4).text())-1
            boundary = ET.SubElement(boundaryCond,'Boundary')
            boundary.set('name',str(self.tableWidget.verticalHeaderItem(j).text()))
            boundary.set('vx',str(self.tableWidget.item(j,0).text()))
            boundary.set('vy',str(self.tableWidget.item(j,1).text()))
            boundary.set('p',str(self.tableWidget.item(j,2).text()))
            boundary.set('T',str(self.tableWidget.item(j,3).text()))
            if self.tableWidget.cellWidget(j, 5).isChecked():
                if not OF_flag:
                    OutFlow = ET.SubElement(data,'OutFlow')
                OF_flag = True
                OF = ET.SubElement(OutFlow,'OF')
                OF.set('name',str(self.tableWidget.verticalHeaderItem(j).text()))



        initialCond = ET.SubElement(data,'InitialCondition')

        initialCond.set('vx','0')
        initialCond.set('vy','0')
        initialCond.set('p','0')
        initialCond.set('T','0')      

        xml_ = ET.tostring(data)
        xml = minidom.parseString(xml_)
        f = open(save_name, "w")
        f.write(xml.toprettyxml(indent="  "))

        if self.checkBox_particles.isChecked():
            data_p = ET.Element('Case')
            data_p.set('name',os.path.basename(save_name))

            particles = ET.SubElement(data_p,'Particles')

            particles.set('nparticles',self.textEdit_Re_3.toPlainText())
            particles.set('rho',self.textEdit_rho.toPlainText())
            particles.set('diameter',self.textEdit_D.toPlainText())
            particles.set('nLoop',self.textEdit_Re_4.toPlainText())
            particles.set('lim_inf_x',self.textEdit_Ga_2.toPlainText())
            particles.set('lim_sup_x',self.textEdit_Ga_3.toPlainText())
            particles.set('lim_inf_y',self.textEdit_Gr_2.toPlainText())
            particles.set('lim_sup_y',self.textEdit_Gr_3.toPlainText())

            xml_2= ET.tostring(data_p)
            xml2 = minidom.parseString(xml_2)
            f2 = open(os.path.splitext(save_name)[0] + '_particles.xml', "w")
            f2.write(xml2.toprettyxml(indent="  "))

    def loadSimulation(self):
        while self.tableWidget.rowCount() > 0:
           self.tableWidget.removeRow(0)
        self.sim_name,_ = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open file', 'C:','(*.xml)')
        
        path = os.path.abspath(self.sim_name)
        f = open(path, encoding="utf-8")

        root = ET.fromstring(f.read())
        
        par = root.find('Parameters')
        self.textEdit_Re.setText(par.attrib['Re'])
        self.textEdit_Pr.setText(par.attrib['Pr'])
        self.textEdit_Gr.setText(par.attrib['Gr'])
        if par.attrib['Fr_flag'] == 'True':
            if self.checkBox_conv.isChecked():
                self.checkBox_conv.setChecked(False)
        else:
            if not self.checkBox_conv.isChecked():
                self.checkBox_conv.setChecked(True)
                
        if par.attrib['particles'] == 'True':
            self.checkBox_particles.setChecked(True)
            self.particles = True
        else:
            self.checkBox_particles.setChecked(False)
            self.particles = False
            
        if self.checkBox_conv.isChecked():
            self.textEdit_Ga.setText(par.attrib['Ga'])
        else:
            self.textEdit_Ga.setText(par.attrib['Fr'])
        
        
        BC_list = []
        for child in root.find('BoundaryCondition'):
            BC_list.append(child.attrib)
            
        for i in range(len(BC_list)):
            rowPosition = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowPosition)
            item = QtWidgets.QTableWidgetItem(BC_list[i]['name'])
            self.tableWidget.setVerticalHeaderItem(i, item)
            it = QtWidgets.QTableWidgetItem(BC_list[i]['vx'])
            self.tableWidget.setItem(i, 0, it)
            it = QtWidgets.QTableWidgetItem(BC_list[i]['vy'])
            self.tableWidget.setItem(i, 1, it)
            it = QtWidgets.QTableWidgetItem(BC_list[i]['p'])
            self.tableWidget.setItem(i, 2, it)
            it = QtWidgets.QTableWidgetItem(BC_list[i]['T'])
            self.tableWidget.setItem(i, 3, it)
            it = QtWidgets.QTableWidgetItem(str(i+1))
            self.tableWidget.setItem(i, 4, it)
            checkbox = QtWidgets.QCheckBox()
            self.tableWidget.setCellWidget(i, 5, checkbox)
            
        OF_list = []
        if root.find('OutFlow') is not None:
            for child in root.find('OutFlow'):
                OF_list.append(child.attrib['name'])
        
        for i in range (self.tableWidget.rowCount()):
            header = self.tableWidget.verticalHeaderItem(i).text()
            if header in OF_list:
                self.tableWidget.cellWidget(i, 5).setChecked(True)
            else:
                self.tableWidget.cellWidget(i, 5).setChecked(False)

            
        self.tableWidget.setVisible(True)
        if os.path.exists(os.path.dirname(self.sim_name) + '/' + os.path.splitext(root.attrib['name'])[0] + '.msh'):
            self.fname = os.path.dirname(self.sim_name) + '/' + os.path.splitext(root.attrib['name'])[0] + '.msh'
        else:
            self.fname = os.path.splitext(os.path.abspath(root.attrib['name']))[0] + '.msh'
        
        self.textEdit.setText(self.fname)
        self.textEdit.setReadOnly(True)
        self.tableWidget.verticalHeader().setVisible(True)
        
      
        self.label_3.setText("Case loaded!")
        
        if self.particles and os.path.exists(os.path.splitext(self.sim_name)[0] + '_particles.xml'):
            path = os.path.abspath(os.path.splitext(self.sim_name)[0] + '_particles.xml')
            f2 = open(path, 'r')
            root = ET.fromstring(f2.read())
            par =  root.find('Particles').attrib
            self.textEdit_D.setText(par['diameter'])
            self.textEdit_rho.setText(par['rho'])
            self.textEdit_Re_4.setText(par['nLoop'])
            self.textEdit_Re_3.setText(par['nparticles'])
            self.textEdit_Ga_2.setText(par['lim_inf_x'])
            self.textEdit_Ga_3.setText(par['lim_sup_x'])
            self.textEdit_Gr_2.setText(par['lim_inf_y'])
            self.textEdit_Gr_3.setText(par['lim_sup_y'])
        
    def conv_or_exp(self):
        if self.checkBox_conv.isChecked():
            self.label_Ga.setText('Ga:')
        else:
            self.label_Ga.setText('Fr:')
    
    def set_particles(self):
        if self.checkBox_particles.isChecked():
            self.label_D.setVisible(True)
            self.label_8.setVisible(True)
            self.label_Ga_3.setVisible(True)
            self.label_Ga_4.setVisible(True)
            self.label_Ga_5.setVisible(True)
            self.label_Ga_6.setVisible(True)
            self.label_D_2.setVisible(True)
            self.label_D_3.setVisible(True)
            
            self.textEdit_D.setVisible(True)
            self.textEdit_rho.setVisible(True)
            self.textEdit_Ga_2.setVisible(True)
            self.textEdit_Ga_3.setVisible(True)
            self.textEdit_Gr_2.setVisible(True)
            self.textEdit_Gr_3.setVisible(True)
            self.textEdit_Re_3.setVisible(True)
            self.textEdit_Re_4.setVisible(True)
            self.particles = True
        else:
            self.label_D.setVisible(False)
            self.label_8.setVisible(False)
            self.label_Ga_3.setVisible(False)
            self.label_Ga_4.setVisible(False)
            self.label_Ga_5.setVisible(False)
            self.label_Ga_6.setVisible(False)
            self.label_D_2.setVisible(False)
            self.label_D_3.setVisible(False)
            
            self.textEdit_D.setVisible(False)
            self.textEdit_rho.setVisible(False)
            self.textEdit_Ga_2.setVisible(False)
            self.textEdit_Ga_3.setVisible(False)
            self.textEdit_Gr_2.setVisible(False)
            self.textEdit_Gr_3.setVisible(False)
            self.textEdit_Re_3.setVisible(False)
            self.textEdit_Re_4.setVisible(False)
            self.particles = False
            
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "JFlow"))
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "vx"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "vy"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "p"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "T"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Priority"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Out Flow"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.label_2.setText(_translate("MainWindow", "Mesh selection"))
        self.label_4.setText(_translate("MainWindow", "Re:"))
        self.label_5.setText(_translate("MainWindow", "Pr:"))
        self.label_Ga.setText(_translate("MainWindow", "Fr:"))
        self.label_Ga_2.setText(_translate("MainWindow", "Gr:"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.label_6.setText(_translate("MainWindow", "Particles"))
        self.label_D.setText(_translate("MainWindow", "D:"))
        self.label_Ga_3.setText(_translate("MainWindow", "min_x:"))
        self.label_8.setText(_translate("MainWindow", "rho: "))
        self.label_Ga_4.setText(_translate("MainWindow", "min_y:"))
        self.label_Ga_5.setText(_translate("MainWindow", "max_y:"))
        self.label_Ga_6.setText(_translate("MainWindow", "max_x:"))
        self.label_D_2.setText(_translate("MainWindow", "Number:"))
        self.label_D_3.setText(_translate("MainWindow", "Loops:"))
        self.label_7.setText(_translate("MainWindow", "Time step:"))
        self.checkBox_conv.setText(_translate("MainWindow", "Convective"))
        self.checkBox_particles.setText(_translate("MainWindow", "Particles"))
        self.label_9.setText(_translate("MainWindow", "Start iteration:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Simtab), _translate("MainWindow", "Simulation settings"))
        self.pushButton_2.setText(_translate("MainWindow", "XY"))
        self.comboBox.setItemText(0, _translate("MainWindow", "v_x"))
        self.comboBox.setItemText(1, _translate("MainWindow", "v_y"))
        self.comboBox.setItemText(2, _translate("MainWindow", "p"))
        self.comboBox.setItemText(3, _translate("MainWindow", "T"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Outputtab), _translate("MainWindow", "Output"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen_simulation.setText(_translate("MainWindow", "Open simulation settings"))
        self.actionSave_simulation_settings.setText(_translate("MainWindow", "Save simulation settings"))
        
        self.pushButton_3.setText(_translate("MainWindow", "Adjust"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
