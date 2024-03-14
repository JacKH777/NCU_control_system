                        # -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bci_gui.ui'
#
# Created by: PyQt5 U
# I code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!



from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')

COLOR = 'gray'
mpl.rcParams['text.color'] = COLOR
# mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams["font.family"] = "monospace"

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        color = 'steelblue'
        linewidth = 0.8
        time_span = 500

        fig = plt.figure()
        gs = fig.add_gridspec(5, hspace=0.1)
        self.axs = gs.subplots(sharex=False, sharey=False)

        channel_list = ['Voltage\n(V)', 'Desired Position\n(deg)', 'Actual Position\n(deg)', 'Comparison\n(deg)', 'Error\n(deg)']
        colors = ['#ff7e26', '#b65e38', '#62856d', '#484c4d', '#073d51', '#45818e']
        xticks = [x for x in range(0, time_span + 1, 50)]
        xticklabels = [str((time/10)) for time in range(0,  time_span + 1, 50)]        
        
        self.lines = []
        for i in range(5):
                # add line object
                self.lines.append(self.axs[i].plot([], [], c=colors[i], lw=linewidth)[0])

                if i == 0 :
                        # set x, y lim
                        self.axs[i].set_xlim(0, time_span)    
                        self.axs[i].set_ylim(-0.1, 10)
                elif i in [1, 2, 3]:
                        # set x, y lim
                        self.axs[i].set_xlim(0, time_span)    
                        self.axs[i].set_ylim(10, 110)   
                else :                               
                        # set x, y lim
                        self.axs[i].set_xlim(0, time_span)    
                        self.axs[i].set_ylim(-20, 20)  
                # set label fontsize
                self.axs[i].tick_params(axis='both', which='major', labelsize=6)
                self.axs[i].yaxis.get_offset_text().set_fontsize(6)

                # set channel name & position
                self.axs[i].set_ylabel(channel_list[i], fontsize=9, rotation=0)
                self.axs[i].yaxis.set_label_coords(-0.14, 0.32) 

                # set xtick & xticklabel
                self.axs[i].set_xticks(xticks)       
                self.axs[i].set_xticklabels(xticklabels)

                # # set grid 
                # self.axs[i].set_yticks([-4e-5, 0, 4e-5], minor=True)
                self.axs[i].grid(axis='y') # 设置 y 就在轴方向显示网格线
                self.axs[i].grid(which="minor",alpha=0.3)

        line2, = self.axs[3].plot([], [], c=colors[5], lw=linewidth)
        self.lines.append(line2)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in self.axs:
            ax.label_outer()

        fig.subplots_adjust(0.2, 0.05, 0.99, 0.97) # left, bottom, right, top 

        super(MplCanvas, self).__init__(fig)




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 481)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 201, 191))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.btnCon = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnCon.sizePolicy().hasHeightForWidth())
        self.btnCon.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.btnCon.setFont(font)
        self.btnCon.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnCon.setMouseTracking(False)
        self.btnCon.setStyleSheet("QPushButton {\n"
"    background-color: #ffffff;\n"
"    border: 1px solid #dcdfe6;\n"
"    padding: 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #ecf5ff;\n"
"    color: #409eff;\n"
"}")
        self.btnCon.setObjectName("btnCon")
        self.gridLayout.addWidget(self.btnCon, 0, 0, 1, 1)
        self.btnSave = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSave.sizePolicy().hasHeightForWidth())
        self.btnSave.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.btnSave.setFont(font)
        self.btnSave.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnSave.setStyleSheet("QPushButton {\n"
"    background-color: #ffffff;\n"
"    border: 1px solid #dcdfe6;\n"
"    padding: 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #d9ead3;\n"
"    color: #198c19;\n"
"}")
        self.btnSave.setObjectName("btnSave")
        self.gridLayout.addWidget(self.btnSave, 1, 0, 1, 1)
        ##
        self.btnSavePic = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSavePic.sizePolicy().hasHeightForWidth())
        self.btnSavePic.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.btnSavePic.setFont(font)
        self.btnSavePic.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnSavePic.setMouseTracking(False)
        self.btnSavePic.setStyleSheet("QPushButton {\n"
"    background-color: #ffffff;\n"
"    border: 1px solid #dcdfe6;\n"
"    padding: 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #ecf5ff;\n"
"    color: #409eff;\n"
"}")
        self.btnSavePic.setObjectName("btnSavePic")
        self.gridLayout.addWidget(self.btnSavePic, 3, 0, 1, 1)
        self.btnSavePic = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSavePic.sizePolicy().hasHeightForWidth())
        self.btnSavePic.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.btnSavePic.setFont(font)
        self.btnSavePic.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnSavePic.setStyleSheet("QPushButton {\n"
"    background-color: #ffffff;\n"
"    border: 1px solid #dcdfe6;\n"
"    padding: 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #d9ead3;\n"
"    color: #198c19;\n"
"}")
        self.btnSavePic.setObjectName("btnSavePic")
        self.gridLayout.addWidget(self.btnSavePic, 3, 0, 1, 1)
        ##
        self.btnDisCon = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnDisCon.sizePolicy().hasHeightForWidth())
        self.btnDisCon.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.btnDisCon.setFont(font)
        self.btnDisCon.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnDisCon.setStyleSheet("QPushButton {\n"
"    background-color: #ffffff;\n"
"    border: 1px solid #dcdfe6;\n"
"    padding: 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color:#f4cccc;\n"
"    color: #F44336;\n"
"}")
        self.btnDisCon.setObjectName("btnDisCon")
        self.gridLayout.addWidget(self.btnDisCon, 2, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.comboBox.setFont(font)
        self.comboBox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox.setStyleSheet("QComboBox {\n"
"    border: 1px solid #dcdfe6;\n"
"    border-radius: 3px;\n"
"    padding: 1px 2px 1px 2px;  \n"
"    min-width: 9em;   \n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"     border: 0px; \n"
"}\n"
"\n"
"QComboBox:hover {\n"
"    background-color: #F0F0F0;\n"
"    color: #0A4D68;\n"
"}\n"
"\n"
"")
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 5, 0, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(220, 10, 741, 461))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.canvas = MplCanvas()
        self.canvas.setStyleSheet("background-color: #ffffff;\n"
"padding: 10px;\n"
"border: 1px solid #dcdfe6;\n"
"border-radius: 5px;\n"
"")
        self.canvas.setObjectName("canvas")
        self.verticalLayout_3.addWidget(self.canvas)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 209, 201, 261))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.message = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.message.sizePolicy().hasHeightForWidth())
        self.message.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.message.setFont(font)
        self.message.setStyleSheet("border: 1px solid #dcdfe6;\n"
"border-radius: 5px;\n"
"background-color: rgb(250, 250, 250);")
        self.message.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.message.setObjectName("message")
        self.verticalLayout.addWidget(self.message)
        # self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        # font = QtGui.QFont()
        # font.setFamily("Consolas")
        # self.label_3.setFont(font)
        # self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_3.setObjectName("label_3")
        # self.verticalLayout.addWidget(self.label_3)
#      self.label_time = QtWidgets.QLabel(self.verticalLayoutWidget_2)
#         self.label_time.setEnabled(True)
#         sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
#         sizePolicy.setHorizontalStretch(0)
#         sizePolicy.setVerticalStretch(0)
#         sizePolicy.setHeightForWidth(self.label_time.sizePolicy().hasHeightForWidth())
#         self.label_time.setSizePolicy(sizePolicy)
#         font = QtGui.QFont()
#         font.setFamily("Consolas")
#         self.label_time.setFont(font)
#         self.label_time.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
#         self.label_time.setStyleSheet("border: 1px solid #dcdfe6;\n"
# "border-radius: 5px;\n"
# "background-color: rgb(250, 250, 250);")
#         self.label_time.setText("")
#         self.label_time.setAlignment(QtCore.Qt.AlignCenter)
#         self.label_time.setObjectName("label_time")
#         self.verticalLayout.addWidget(self.label_time)
#         self.label_3.raise_()
#         self.label_time.raise_()
        self.message.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "COM Port"))
        self.btnCon.setText(_translate("MainWindow", "Connect"))
        self.btnSave.setText(_translate("MainWindow", "Show Data"))
        self.btnDisCon.setText(_translate("MainWindow", "Disconnect"))
        self.btnSavePic.setText(_translate("MainWindow", "Save Data Pic"))
        self.message.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        #self.label_3.setText(_translate("MainWindow", "Elapsed Time"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

