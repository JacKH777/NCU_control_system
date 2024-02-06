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

from smc_system_detail import control_system,Control,return_simulation_pma_angle

import skfuzzy as fuzz
import skfuzzy.control as ctrl
from fuzzy_neural_principle import fuzzy_system,RealTimeGaussianPlot,self_fuzzy_system


# https://www.pythonguis.com/tutorials/plotting-matplotlib/
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# from EEGModels import EEGNet
from scipy import signal

import serial.tools.list_ports

from decoder_function import decoder


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
        if len(self.desire_deg_array) >= 500: 
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

        self.canvas.lines[4].set_data(xdat, y_data_1 - ydat)

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

        self.triangle_angle = [30,31,31,32,32,33,34,34,35,35,36,37,37,38,38,39,40,40,41,41,42,43,43,44,44,45,46,46,47,47,48,49,49,50,51,51,52,52,53,54,54,55,55,56,57,57,58,58,59,60,60,61,61,62,63,63,64,64,65,66,66,67,67,68,69,69,70,70,71,72,72,73,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,84,84,85,85,86,87,87,88,88,89,90,90,89,88,88,87,87,86,85,85,84,84,83,82,82,81,81,80,79,79,78,78,77,76,76,75,75,74,73,73,72,72,71,70,70,69,69,68,67,67,66,66,65,64,64,63,63,62,61,61,60,60,59,58,58,57,57,56,55,55,54,54,53,52,52,51,51,50,49,49,48,47,47,46,46,45,44,44,43,43,42,41,41,40,40,39,38,38,37,37,36,35,35,34,34,33,32,32,31,31,30]
        self.triangle_angle_voltage = [5700,5881,6062,6243,6424,6605,6785,6966,7147,7328,7509,7690,7871,8052,8233,8414,8594,8775,8956,9137,9318,9499,9680,9861,10042,10223,10404,10584,10765,10946,11127,11308,11489,11670,11851,12032,12213,12393,12574,12755,12936,13117,13298,13479,13660,13841,14022,14203,14383,14564,14745,14926,15107,15288,15469,15650,15831,16012,16192,16373,16554,16735,16916,17097,17278,17459,17640,17821,18002,18182,18363,18544,18725,18906,19087,19268,19449,19630,19811,19991,20172,20353,20534,20715,20896,21077,21258,21439,21620,21801,21981,22162,22343,22524,22705,22886,23067,23248,23429,23610,23610,23429,23248,23067,22886,22705,22524,22343,22162,21981,21801,21620,21439,21258,21077,20896,20715,20534,20353,20172,19991,19811,19630,19449,19268,19087,18906,18725,18544,18363,18182,18002,17821,17640,17459,17278,17097,16916,16735,16554,16373,16192,16012,15831,15650,15469,15288,15107,14926,14745,14564,14383,14203,14022,13841,13660,13479,13298,13117,12936,12755,12574,12393,12213,12032,11851,11670,11489,11308,11127,10946,10765,10584,10404,10223,10042,9861,9680,9499,9318,9137,8956,8775,8594,8414,8233,8052,7871,7690,7509,7328,7147,6966,6785,6605,6424,6243,6062,5881,5700]
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
        print(f"Successfull Open  {COM_PORT}")
        queue_gui_message.put(f"Successfull Open")

        # self.ser_2 = serial.Serial('COM4', 115200)
        right_hand = decoder()
        right_hand.get_com_port('COM4')
        print(f"Successfull Open COM4")

        C = Control()

        # 模擬模式=True, 步階響應模式 mode=1
        simulation = False
        mode = 0

        Idx = 0
        test = 0

        desire_angle = self.triangle_angle[0]
        actual_angle = self.triangle_angle[0]
        learning_array = [0] * 500
        first_period = True
        controller_u = 0.8
        controller_u_output = 0

        smc_lambda = 0.1   # 0.2 越快到滑膜面
        k_l1 = 0.05          # 0.5
        k_l2 = 0.02  

        first_count = 0
        reset_count = 0

        last_error = 0
        new_u = 0
        ########old1
        #a = fuzzy_system()
        ##############
        fuzzy2 = self_fuzzy_system()


        # plt.ion()
        plotter = RealTimeGaussianPlot()
        plt.show(block=False)
        plotter.set_data()
        total_error = 0
        while True:

            # 前 (test/10) 秒不作動
            if test < 100:
                Idx = 0
            else:
                Idx = Idx + 1
                if Idx == 200:
                    Idx = 0 

            test = test + 1

            ####################### 目標路徑 #######################
            desire_angle = self.triangle_angle[Idx]
            if mode == 1:
                if desire_angle > 55 or test > 100:
                    desire_angle = 50
                else :
                    desire_angle = 30
            #######################################################

            ########### Decoder(真實回饋，simulation=False) #########
            # if simulation == False:
            #     actual_angle = right_hand.get_angle()
                # self.ser_2.write(b'\x54')
                # read_data = self.ser_2.read(2)
                # received_val = int.from_bytes(read_data, byteorder='little')
        
                # # 将整数转换成二进制，并移除最高两位
                # binary_val = bin(received_val)[2:].zfill(16)  # 将整数转换为16位的二进制字符串
                # truncated_binary = binary_val[2:]  # 移除最高两位
                # actual_angle = int(truncated_binary, 2)

                # # 校正
                # if first_count==0:
                #     reset_count = actual_angle
                #     first_count = 1
                # if actual_angle < reset_count:
                #     actual_angle = ((actual_angle-reset_count+16384)/16383*360)+20
                # else:
                #     actual_angle = ((actual_angle-reset_count)/16383*360)+20
                # if actual_angle > 350:
                #     actual_angle = 20

            ########################################################
            
            # 控制系統
            #
            # controller_u, learning_array, first_period, C = control_system(controller_u,desire_angle,actual_angle,learning_array,Idx,first_period, C, smc_lambda, k_l1, k_l2)
            #
            #########取樣增加
            # for i in range(3):
            #     if i == 0:
            #         error = 0
            #         delta = 0
            #     actual_angle = right_hand.get_angle()
            #     error = error + desire_angle - actual_angle
            #     delta = delta + error -  last_error
            #     if i == 2:
            #         error = error/3
            #         delta = delta/3
            #####################
            ###########old1
            # a.restart_system()
            # output_m,new_u = a.calculate(error,delta)
            # new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width = a.output_gauss_learning(error,delta,0.00005,0.00005,0.00005,output_m)
            # plotter.set_data(new_error_gauss_center, new_error_gauss_width)
            # a.new_output_gauss(new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width)
            # new_u = new_u * 0.02
            # controller_u = controller_u + new_u
            # print(new_error_gauss_width)
            # last_error = error
            ######################
                    
            ########new need
            actual_angle = right_hand.get_angle()
            error = desire_angle - actual_angle
            delta = error -  last_error
            new_u,c, w = fuzzy2.fuzzy_rule(error, delta)
            # plotter.set_data(c, w)
            new_u = new_u * 0.003
            controller_u = controller_u + new_u
            last_error = error
            ##########
                    
            # #no control
            # controller_u_output = self.triangle_angle_voltage[Idx]
            # actual_angle = right_hand.get_angle()
            # error = desire_angle - actual_angle
            # queue_receive_deg.put(actual_angle)
            # queue_desire_deg.put(desire_angle)
            # queue_voltage.put(controller_u_output/65535*10)
            # #

            # 儲存結果 need
            queue_receive_deg.put(actual_angle)
            queue_desire_deg.put(desire_angle)
            queue_voltage.put(controller_u)
            # 轉成 16 bits 電壓值 need
            controller_u_output = controller_u/10*65535
            controller_u_output = int(controller_u_output)
            
            if simulation == False:

                # if controller_u_output>45000:
                #     controller_u_output = 45000
                if controller_u_output < 0:
                    controller_u_output = 0

            if actual_angle > 100:
                controller_u_output = 0
                self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
                break
            
            self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))

            if Idx == 199:
                print(total_error)
                total_error = 0
            else:
                total_error = total_error+(error**2)

            if simulation == True:
                actual_angle = return_simulation_pma_angle(self.df_pma_angle,controller_u_output,actual_angle)

            time.sleep(0.02) #delay 0.1 sec



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())