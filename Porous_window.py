from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Porous_window(object):
    def setupUi(self, Porous_window):
        Porous_window.setObjectName("Porous_window")
        Porous_window.setFixedSize(491, 368)
        self.centralwidget = QtWidgets.QWidget(Porous_window)
        self.centralwidget.setObjectName("centralwidget")
        self.save_Button = QtWidgets.QPushButton(self.centralwidget)
        self.save_Button.setGeometry(QtCore.QRect(130, 270, 93, 28))
        self.save_Button.setObjectName("save_Button")
        self.cancel_Button = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_Button.setGeometry(QtCore.QRect(280, 270, 93, 28))
        self.cancel_Button.setObjectName("cancel_Button")
        self.darcy_label = QtWidgets.QLabel(self.centralwidget)
        self.darcy_label.setGeometry(QtCore.QRect(110, 40, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.darcy_label.setFont(font)
        self.darcy_label.setObjectName("darcy_label")
        self.darcy_input = QtWidgets.QTextEdit(self.centralwidget)
        self.darcy_input.setGeometry(QtCore.QRect(140, 30, 81, 31))
        self.darcy_input.setObjectName("darcy_input")
        self.Fo_input = QtWidgets.QTextEdit(self.centralwidget)
        self.Fo_input.setGeometry(QtCore.QRect(280, 30, 81, 31))
        self.Fo_input.setObjectName("Fo_input")
        self.Fo_label = QtWidgets.QLabel(self.centralwidget)
        self.Fo_label.setGeometry(QtCore.QRect(250, 40, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Fo_label.setFont(font)
        self.Fo_label.setObjectName("Fo_label")

        Porous_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Porous_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 491, 26))
        self.menubar.setObjectName("menubar")
        Porous_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Porous_window)
        self.statusbar.setObjectName("statusbar")
        Porous_window.setStatusBar(self.statusbar)

        self.retranslateUi(Porous_window)
        QtCore.QMetaObject.connectSlotsByName(Porous_window)

    def retranslateUi(self, Porous_window):
        _translate = QtCore.QCoreApplication.translate
        Porous_window.setWindowTitle(_translate("Porous_window", "Porous regions"))
        self.save_Button.setText(_translate("Porous_window", "Save"))
        self.cancel_Button.setText(_translate("Porous_window", "Cancel"))
        self.darcy_label.setText(_translate("Porous_window", "Da:"))
        self.Fo_label.setText(_translate("Porous_window", "Fo:"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Porous_window = QtWidgets.QMainWindow()
    ui = Ui_Porous_window()
    ui.setupUi(Porous_window)
    Porous_window.show()
    sys.exit(app.exec_())
