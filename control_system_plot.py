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

import tensorflow as tf

from smc_system_detail import control_system,Control,return_simulation_pma_angle

import skfuzzy as fuzz
import skfuzzy.control as ctrl
from fuzzy_neural_principle import fuzzy_system,RealTimeGaussianPlot,self_fuzzy_system
from keras_fuzzy import ANFIS


# https://www.pythonguis.com/tutorials/plotting-matplotlib/
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# from EEGModels import EEGNet
from scipy import signal

import serial.tools.list_ports

from decoder_function import decoder

TF_ENABLE_ONEDNN_OPTS=0


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
        self.btnCon.clicked.connect(self.ShowData)  # 顯示
        #self.btnSave.clicked.connect(self.ShowData)  # 顯示
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

        self.triangle_angle = [30,30,30,30,30,30,30,31,31,31,32,32,33,34,34,35,36,36,37,38,39,39,40,41,42,42,43,44,45,45,46,47,48,48,49,50,51,51,52,53,54,54,55,56,56,57,58,59,59,60,61,62,62,63,64,65,65,66,67,68,68,69,70,71,71,72,73,74,74,75,76,77,77,78,79,79,80,81,82,82,83,84,85,85,86,87,88,88,89,89,90,90,91,91,92,92,92,92,92,92,92,92,92,92,92,92,92,91,91,90,90,89,89,88,88,87,86,85,85,84,83,82,82,81,80,79,79,78,77,77,76,75,74,74,73,72,71,71,70,69,68,68,67,66,65,65,64,63,62,62,61,60,59,59,58,57,56,56,55,54,54,53,52,51,51,50,49,48,48,47,46,45,45,44,43,42,42,41,40,39,39,38,37,36,36,35,34,34,33,32,32,31,31,31,30,30,30,30,30,30]
        self.triangle_angle_voltage = [3100,3235,3369,3504,3639,3773,3908,4043,4177,4312,4447,4581,4716,4851,4985,5120,5255,5389,5524,5659,5793,5928,6063,6197,6332,6467,6602,6736,6871,7006,7140,7275,7410,7544,7679,7814,7948,8083,8218,8352,8487,8622,8756,8891,9026,9160,9295,9430,9564,9699,9834,9968,10103,10238,10372,10507,10642,10776,10911,11046,11180,11315,11450,11584,11719,11854,11988,12123,12258,12392,12527,12662,12796,12931,13066,13201,13335,13470,13605,13739,13874,14009,14143,14278,14413,14547,14682,14817,14951,15086,15221,15355,15490,15625,15759,15894,16029,16163,16298,16433,16433,16298,16163,16029,15894,15759,15625,15490,15355,15221,15086,14951,14817,14682,14547,14413,14278,14143,14009,13874,13739,13605,13470,13335,13201,13066,12931,12796,12662,12527,12392,12258,12123,11988,11854,11719,11584,11450,11315,11180,11046,10911,10776,10642,10507,10372,10238,10103,9968,9834,9699,9564,9430,9295,9160,9026,8891,8756,8622,8487,8352,8218,8083,7948,7814,7679,7544,7410,7275,7140,7006,6871,6736,6602,6467,6332,6197,6063,5928,5793,5659,5524,5389,5255,5120,4985,4851,4716,4581,4447,4312,4177,4043,3908,3773,3639,3504,3369,3235,3100]
        
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
        controller_u = 0.65
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
        error = 0
        delta = 0
        fis = ANFIS(n_inputs=2, n_rules=7, learning_rate=0.1)
        early_stop = 0
        last_total_error = 100 #max

        
        total_duration = 0.05

        # plotter = RealTimeGaussianPlot()
        # plt.show(block=False)
        # plotter.set_data()
        total_error = 0

        #keras
        while True:
            # 记录开始时间
            start_time = time.time()
            # 前 (test/10) 秒不作動
            if test <4/total_duration:
                Idx = 0
            else:
                Idx = Idx + 1
                if Idx == 200:
                    Idx = 0 

            test = test + 1

            ####################### 目標路徑 #######################
            desire_angle = self.triangle_angle[Idx]
            if mode == 1:
                if desire_angle > 55 :
                    desire_angle = 50
                else :
                    desire_angle = 30
            #######################################################

            
            ########new need
            actual_angle = right_hand.get_angle()
            error = desire_angle - actual_angle
            delta = (error -  last_error)
            if early_stop == 0:
                new_u, mu_error, sigma_error, mu_delta, sigma_delta, y= fis.train([error],[delta], [desire_angle],[actual_angle])
            # else:
            #     new_u = fis.predict([error],[delta],[actual_angle],last_mu_error, last_sigma_error,last_mu_delta, last_sigma_delta, last_y)
            # new_u,c, w = fuzzy2.fuzzy_rule(error, delta)
            # plotter.set_data(c, w)
            new_u = new_u * 0.0035
            controller_u = controller_u + new_u
            # print(controller_u)
            last_error = error
            # print(error,delta)
            # print(c,w)
            ##########

            # 儲存結果 need
            if test < 4/total_duration:
                actual_angle = desire_angle
                total_error = 0
            int_actual_angle = int(actual_angle)
            queue_receive_deg.put(int_actual_angle)
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
                total_error = (total_error/200)**0.5
                print(total_error)
                # if total_error <= last_total_error:
                #     last_mu_error, last_sigma_error,last_mu_delta, last_sigma_delta, last_y = mu_error, sigma_error, mu_delta, sigma_delta, y
                #     last_total_error = total_error
                # else:
                #     early_stop = 1
                total_error = 0
            else:
                total_error = total_error+(error**2)

            if simulation == True:
                actual_angle = return_simulation_pma_angle(self.df_pma_angle,controller_u_output,actual_angle)

            # time.sleep(system_time) #delay 0.1 sec
            elapsed_time = time.time() - start_time

            # 如果运行时间小于指定的总时间，则补充剩余的时间
            if elapsed_time < total_duration:
                remaining_time = total_duration - elapsed_time
                time.sleep(remaining_time)


        # while True:
        #     # 记录开始时间
        #     start_time = time.time()
        #     # 前 (test/10) 秒不作動
        #     if test <5/total_duration:
        #         Idx = 0
        #     else:
        #         Idx = Idx + 1
        #         if Idx == 200:
        #             Idx = 0 

        #     test = test + 1

        #     ####################### 目標路徑 #######################
        #     desire_angle = self.triangle_angle[Idx]
        #     if mode == 1:
        #         if desire_angle > 55 or test > 100:
        #             desire_angle = 50
        #         else :
        #             desire_angle = 30
        #     #######################################################

        #     ########### Decoder(真實回饋，simulation=False) #########
        #     # if simulation == False:
        #     #     actual_angle = right_hand.get_angle()
        #         # self.ser_2.write(b'\x54')
        #         # read_data = self.ser_2.read(2)
        #         # received_val = int.from_bytes(read_data, byteorder='little')
        
        #         # # 将整数转换成二进制，并移除最高两位
        #         # binary_val = bin(received_val)[2:].zfill(16)  # 将整数转换为16位的二进制字符串
        #         # truncated_binary = binary_val[2:]  # 移除最高两位
        #         # actual_angle = int(truncated_binary, 2)

        #         # # 校正
        #         # if first_count==0:
        #         #     reset_count = actual_angle
        #         #     first_count = 1
        #         # if actual_angle < reset_count:
        #         #     actual_angle = ((actual_angle-reset_count+16384)/16383*360)+20
        #         # else:
        #         #     actual_angle = ((actual_angle-reset_count)/16383*360)+20
        #         # if actual_angle > 350:
        #         #     actual_angle = 20

        #     ########################################################
            
        #     # 控制系統
        #     #
        #     # controller_u, learning_array, first_period, C = control_system(controller_u,desire_angle,actual_angle,learning_array,Idx,first_period, C, smc_lambda, k_l1, k_l2)
        #     #
        #     #########取樣增加
        #     # for i in range(3):
        #     #     if i == 0:
        #     #         error = 0
        #     #         delta = 0
        #     #     actual_angle = right_hand.get_angle()
        #     #     error = error + desire_angle - actual_angle
        #     #     delta = delta + error -  last_error
        #     #     if i == 2:
        #     #         error = error/3
        #     #         delta = delta/3
        #     #####################
        #     ###########old1
        #     # a.restart_system()
        #     # output_m,new_u = a.calculate(error,delta)
        #     # new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width = a.output_gauss_learning(error,delta,0.00005,0.00005,0.00005,output_m)
        #     # plotter.set_data(new_error_gauss_center, new_error_gauss_width)
        #     # a.new_output_gauss(new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width)
        #     # new_u = new_u * 0.02
        #     # controller_u = controller_u + new_u
        #     # print(new_error_gauss_width)
        #     # last_error = error
        #     ######################
                    
        #     ########new need
        #     new_u,c, w = fuzzy2.fuzzy_rule(error, delta)
        #     plotter.set_data(c, w)
        #     new_u = new_u * 0.01
        #     controller_u = controller_u + new_u
        #     actual_angle = right_hand.get_angle()
        #     error = desire_angle - actual_angle
        #     delta = error -  last_error
        #     fuzzy2.output_gauss_learning(error,delta)
        #     last_error = error
        #     print(error,delta)
        #     # print(c,w)
        #     ##########
                    
        #     # #no control
        #     # controller_u_output = self.triangle_angle_voltage[Idx]
        #     # actual_angle = right_hand.get_angle()
        #     # error = desire_angle - actual_angle
        #     # queue_receive_deg.put(actual_angle)
        #     # queue_desire_deg.put(desire_angle)
        #     # queue_voltage.put(controller_u_output/65535*10)
        #     # #

        #     # 儲存結果 need
        #     if test < 5/total_duration:
        #         actual_angle = desire_angle
        #         total_error = 0
        #     int_actual_angle = int(actual_angle)
        #     queue_receive_deg.put(int_actual_angle)
        #     queue_desire_deg.put(desire_angle)
        #     queue_voltage.put(controller_u)
        #     # 轉成 16 bits 電壓值 need
        #     controller_u_output = controller_u/10*65535
        #     controller_u_output = int(controller_u_output)
            

        #     if simulation == False:

        #         # if controller_u_output>45000:
        #         #     controller_u_output = 45000
        #         if controller_u_output < 0:
        #             controller_u_output = 0

        #     if actual_angle > 100:
        #         controller_u_output = 0
        #         self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
        #         break
            
        #     self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))

        #     if Idx == 199:
        #         total_error = (total_error/200)**0.5
        #         print(total_error)
        #         total_error = 0
        #     else:
        #         total_error = total_error+(error**2)

        #     if simulation == True:
        #         actual_angle = return_simulation_pma_angle(self.df_pma_angle,controller_u_output,actual_angle)

        #     # time.sleep(system_time) #delay 0.1 sec
        #     elapsed_time = time.time() - start_time

        #     # 如果运行时间小于指定的总时间，则补充剩余的时间
        #     if elapsed_time < total_duration:
        #         remaining_time = total_duration - elapsed_time
        #         time.sleep(remaining_time)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())