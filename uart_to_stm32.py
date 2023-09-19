import serial
import time

# def send_16bit_data_to_stm32(data):
#     ser.write(data.to_bytes(2, byteorder='big'))

def test_uart_sine_wave():
    ser = serial.Serial('COM9', 115200)
    Sine16bit = [
        18000,18890,19778,20662,21539,22407,23265,24110,24940,25753,26546,27319,28070,28795,29494,30165,30806,31416,31993,32536,33043,33514,33946,34340,34693,35006,35277,35505,35691,35834,35933,35988,35999,35966,35889,35768,35604,35396,35146,34854,34521,34148,33735,33283,32794,32269,31709,31115,30490,29833,29148,28436,27697,26936,26152,25348,24527,23689,22838,21974,21101,20221,19335,18445,17555,16665,15779,14899,14026,13162,12311,11473,10652,9848,9064,8303,7564,6852,6167,5510,4885,4291,3731,3206,2717,2265,1852,1479,1146,854,604,396,232,111,34,1,12,67,166,309,495,723,994,1307,1660,2054,2486,2957,3464,4007,4584,5194,5835,6506,7205,7930,8681,9454,10247,11060,11890,12735,13593,14461,15338,16222,17110,18000
    ]

    Idx = 0

    while True:
        val = Sine16bit[Idx]
        Idx = Idx + 1
        if Idx == 128:
            Idx = 0
        ser.write(val.to_bytes(2, byteorder='big'))

        time.sleep(1/10) #delay 0.01 sec

def test_uart_triangle_wave():
    ser = serial.Serial('COM9', 115200)
    Sine16bit = [
        25000
    ]

    Idx = 0

    while True:
        val = Sine16bit[Idx]
        Idx = Idx + 1
        if Idx == 1:
            Idx = 0
        ser.write(val.to_bytes(2, byteorder='big'))
        time.sleep(1/10) #delay 0.01 sec

if __name__ == "__main__":
    test_uart_triangle_wave()