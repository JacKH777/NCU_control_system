# 2021/02/04 EEG_button
#  https://stackoverflow.com/a/6981055/6622587
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
# from Ui_NewGUI import Ui_MainWindow
# from trash import Ui_MainWindow
from control_system_gui import Ui_MainWindow

import sys
import multiprocessing
import serial
import time
import numpy as np
from datetime import datetime
from datetime import datetime
from multiprocessing import Queue

import pandas as pd
from scipy.interpolate import interp1d

from control_system_detail import control_system,Control,return_simulation_pma_angle



# https://www.pythonguis.com/tutorials/plotting-matplotlib/
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# from EEGModels import EEGNet
from scipy import signal

import serial.tools.list_ports


def styled_text(text=None, color="#999999"):
    if text is None:
        text = datetime.now().strftime("%H:%M:%S")                
    return f"<span style=\" font-size:8pt; color:{color};\" >" + text + "</span>"    


class MyMainWindow(QtWidgets. QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Control System')
        
        # 按鍵功能
        self.btnCon.clicked.connect(self.StartConnection)  # 連線
        self.btnDisCon.clicked.connect(self.Disconnection)  # 斷線
        self.btnSave.clicked.connect(self.ShowData)  # 顯示
        self.btnSavePic.clicked.connect(self.SavedataPic)  # 存檔

        self.ser = None

        # 建立資料接收class
        self.dt = DataReceiveThreads() 
        # self.queue_data_save_flag = Queue()
        self.queue_voltage = Queue()
        self.queue_comport = Queue()
        self.queue_gui_message = Queue()
        self.queue_receive_deg = Queue()
        self.queue_desire_deg = Queue()
        self.queue_control_value = Queue()

        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, 
                                                      args=(self.queue_comport, self.queue_voltage, 
                                                             self.queue_gui_message,
                                                            self.queue_receive_deg, self.queue_desire_deg,
                                                            self.queue_control_value
                                                            )) 
        self.raw_total = np.array([])
        self.raw_total_deg = np.array([])
        self.desire_deg_array = np.array([])

        # ------------------------------------ #
        # Show all COM Port in combobox 
        # ------------------------------------ #
        default_idx = -1
        ports = serial.tools.list_ports.comports()
        for i, port in enumerate(ports):
            # port.device = 'COMX'
            if "透過" in port.description:
                default_idx = i
                self.queue_comport.put(port.device)
                print(f"Selected default COM : {port.description}")

                self.message.append(styled_text())                
                self.message.append(f'>> Default COM : {port.device}')

            self.comboBox.addItem(port.device + ' - ' + port.description)
        
        self.comboBox.setCurrentIndex(default_idx)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)

        self.timer_activate = False
    
    def SavedataPic(self):
        current_time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
        filename = '{}_control_pic.png'.format(current_time)
        self.canvas.figure.savefig(filename)
        self.message.append(f'>> Save data pic success')


    def update_plot(self):
        raw = self.queue_voltage.get()
        raw_deg = self.queue_receive_deg.get()
        desire_deg = self.queue_desire_deg.get()
             
        # clear the queue
        while not self.queue_voltage.empty():
            temp = self.queue_voltage.get() 
            del temp
        if len(self.raw_total) >= 500: 
            self.raw_total = self.raw_total[-1:]
            self.raw_total = np.append(self.raw_total, raw)
        else:
            self.raw_total = np.append(self.raw_total, raw)
        #
        while not self.queue_receive_deg.empty():
            temp = self.queue_receive_deg.get() 
            del temp
        if len(self.raw_total_deg) >= 500: 
            self.raw_total_deg = self.raw_total_deg[-1:]
        else:
            self.raw_total_deg = np.append(self.raw_total_deg, raw_deg)
        #
        
        #
        while not self.queue_desire_deg.empty():
            temp = self.queue_desire_deg.get() 
            del temp
        if len(self.desire_deg_array) >= 600: 
            self.desire_deg_array = self.desire_deg_array[-1:]
        else:
            self.desire_deg_array = np.append(self.desire_deg_array, desire_deg)
        #

        ydata = self.raw_total      
        xdata = np.arange(ydata.shape[0])
        self.canvas.lines[0].set_data(xdata, ydata)

        y_data_1 = self.desire_deg_array     
        x_data_1 = np.arange(y_data_1.shape[0])
        self.canvas.lines[1].set_data(x_data_1, y_data_1)

        ydat = self.raw_total_deg     
        xdat = np.arange(ydat.shape[0])
        self.canvas.lines[2].set_data(xdat, ydat)

        self.canvas.lines[3].set_data(x_data_1, y_data_1)
        self.canvas.lines[5].set_data(xdat, ydat)

        self.canvas.lines[4].set_data(xdat, y_data_1-ydat)

        self.canvas.draw()

    def on_combobox_changed(self, index):
        if index < 0:
            return
        # 取得選擇的 COM Port
        COM_PORT = self.comboBox.itemText(index).split(' ')[0]
        print(f'Selected Port: {COM_PORT}')
        self.queue_comport.put(COM_PORT)


        self.message.append(styled_text()) 
        self.message.append(f'>> Selected Port: {COM_PORT}')

    def StartConnection(self):  
        #連線        
        self.multipDataRecv.start()
        # self.queue_data_save_flag.put(False)


        while True:
            if not self.queue_gui_message.empty():
                # Get last selected COM port name from queue
                message = self.queue_gui_message.get()

                self.message.append(styled_text()) 
                self.message.append(f'>> {message}')                
                break
        
        
    def Disconnection(self):

        if not self.multipDataRecv.is_alive():            
            print ("Process has not started")
        else:
            self.message.append(styled_text()) 
            self.message.append(f'>> You can close this window.')
            print("You can close this window")
            self.multipDataRecv.terminate()
            # self.queue_data_save_flag.put(False)
        if self.timer_activate:
            self.timer.stop()
    
    def ShowData(self):
        self.message.append(styled_text())      
        self.message.append(f'>> Start Showing')
        # self.queue_data_save_flag.put(True)
        
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)   
        self.timer_activate = True   

class DataReceiveThreads(Ui_MainWindow):
    def __init__(self):
        self.ser_1 = None
        self.ser_2 = None
        self.Sine16bit = [
        20,21,21,22,23,23,24,25,25,26,27,28,28,29,30,30,31,32,32,33,34,34,35,36,36,37,38,38,39,40,41,41,42,43,43,44,45,45,46,47,47,48,49,49,50,51,51,52,53,53,54,55,56,56,57,58,58,59,60,60,61,62,62,63,64,64,65,66,66,67,68,69,69,70,71,71,72,73,73,74,75,75,76,77,77,78,79,79,80,81,82,82,83,84,84,85,86,86,87,88,88,87,86,86,85,84,84,83,82,82,81,80,79,79,78,77,77,76,75,75,74,73,73,72,71,71,70,69,69,68,67,66,66,65,64,64,63,62,62,61,60,60,59,58,58,57,56,56,55,54,53,53,52,51,51,50,49,49,48,47,47,46,45,45,44,43,43,42,41,41,40,39,38,38,37,36,36,35,34,34,33,32,32,31,30,30,29,28,28,27,26,25,25,24,23,23,22,21,21,20   
]   
        self.triangle_angle = [20,21,23,24,26,27,28,30,31,33,34,36,37,38,40,41,43,44,45,47,48,50,51,53,54,55,57,58,60,61,62,64,65,67,68,69,71,72,74,75,77,78,79,81,82,84,85,86,88,89,89,88,86,85,84,82,81,79,78,77,75,74,72,71,69,68,67,65,64,62,61,60,58,57,55,54,53,51,50,48,47,45,44,43,41,40,38,37,36,34,33,31,30,28,27,26,24,23,21,20]

        excel_file = pd.ExcelFile('PMA_angle.xlsx')
        self.df_pma_angle = excel_file.parse('Sheet1', usecols="B:C", header=None,nrows=200)

    def data_recv(self, queue_comport, queue_voltage, queue_gui_message,queue_receive_deg, queue_desire_deg, queue_control_value):

        while True:            
            if not queue_comport.empty():
                # Get last selected COM port name from queue
                COM_PORT = queue_comport.get()
                break

        print(f"Open {COM_PORT}...")
        self.ser_1 = serial.Serial(COM_PORT, 115200)
        print(f"Successfull Open")
        queue_gui_message.put("Successfull Open")
        self.ser_2 = serial.Serial('COM4', 115200)

        C = Control()

        # 模擬模式=True, 步階響應=1
        simulation = True
        mode = 1

        Idx = 0
        test = 0

        desire_angle = self.triangle_angle[0]
        actual_angle = self.triangle_angle[0]
        learning_array = [0] * 100
        first_period = True
        controller_u = 0
        
        first_count = 0
        reset_count = 0

        while True:

            # 前 (test/10) 秒不作動
            if test < 30:
                Idx = 0
            else:
                Idx = Idx + 1
                if Idx == 100:
                    Idx = 0 
            test = test + 1

            ####################### 目標路徑 #######################
            desire_angle = self.triangle_angle[Idx]
            if mode == 1:
                if desire_angle > 55 or test > 100:
                    desire_angle = 90
                else :
                    desire_angle = 20
            #######################################################

            ########### Decoder(真實回饋，simulation=False) #########
            if simulation == False:
                self.ser_2.write(b'\x54')
                read_data = self.ser_2.read(2)
                received_val = int.from_bytes(read_data, byteorder='little')
        
                # 将整数转换成二进制，并移除最高两位
                binary_val = bin(received_val)[2:].zfill(16)  # 将整数转换为16位的二进制字符串
                truncated_binary = binary_val[2:]  # 移除最高两位
                actual_angle = int(truncated_binary, 2)

                # 校正
                if first_count==0:
                    reset_count = actual_angle
                    first_count = 1
                if actual_angle < reset_count:
                    actual_angle = ((actual_angle-reset_count+16384)/16383*360)+25
                else:
                    actual_angle = ((actual_angle-reset_count)/16383*360)+25
                if actual_angle > 350:
                    actual_angle = 25
            ########################################################
            
            # 控制系統
            controller_u, learning_array, first_period, C = control_system(controller_u,desire_angle,actual_angle,learning_array,Idx,first_period, C)

            # # test
            # if first_period == False:
            #     if Idx == range(15,55) and Idx != 30:
            #         actual_angle = actual_angle - (20/(abs(30-Idx)))
            #     elif Idx == 30:
            #         actual_angle = actual_angle - 20

            # 儲存結果
            queue_receive_deg.put(actual_angle)
            queue_desire_deg.put(desire_angle)
            queue_voltage.put(controller_u)

            # # 轉成 16 bits 電壓值
            # controller_u = controller_u/10*65535
            # controller_u = int(controller_u)

            if simulation == False:

                if controller_u>32767:
                    controller_u = 32767
                elif controller_u<0:
                    controller_u = 0

                self.ser_1.write(controller_u.to_bytes(2, byteorder='big'))

            #queue_voltage.put(controller_u)
            if simulation == True:
                actual_angle = return_simulation_pma_angle(self.df_pma_angle,int(controller_u/10*65535),actual_angle)

            time.sleep(1/10) #delay 0.1 sec



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())