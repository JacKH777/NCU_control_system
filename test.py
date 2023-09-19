import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QApplication
from PyQt5 import QtCore
import serial

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('調整氣壓大小')
        self.setGeometry(500, 500, 1500, 1000)

        self.label = QLabel('電壓值(0~36000)', self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.button_connect_uart = QPushButton('藍芽連線', self)
        self.button_connect_uart.clicked.connect(self.connect_to_stm32)

        self.button_increase = QPushButton('增加電壓', self)
        self.button_increase.clicked.connect(self.increase_voltage)

        self.button_decrease = QPushButton('減小電壓', self)
        self.button_decrease.clicked.connect(self.decrease_voltage)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_connect_uart)
        layout.addWidget(self.button_increase)
        layout.addWidget(self.button_decrease)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.current_voltage = 0
        self.ser = None 
    
    def connect_to_stm32(self):
        if self.ser is None:
            try:
                self.ser = serial.Serial('COM9', 460800)
                self.button_connect_uart.setText('連線成功')
            except serial.SerialException as e:
                self.button_connect_uart.setText("連接失敗")
        else :
            self.ser.close()
            self.ser = None 
            self.button_connect_uart.setText('藍芽連線')

    def increase_voltage(self):
        self.current_voltage += 100
        if self.current_voltage > 36000:
            self.current_voltage = 36000
        self.update_voltage()

    def decrease_voltage(self):
        self.current_voltage -= 100
        if self.current_voltage < 0:
            self.current_voltage = 0
        self.update_voltage()

    def update_voltage(self):
        self.label.setText(f"電壓值：{self.current_voltage}V")
        self.ser.write(self.current_voltage.to_bytes(2, byteorder='big'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())