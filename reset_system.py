import serial
import time

ser_1 = serial.Serial("COM8", 115200)
controller_u_output = 0
triangle_angle = [0,606,1212,1818,2424,3030,3636,4242,4848,5455,6061,6667,7273,7879,8485,9091,9697,10303,10909,11515,12121,12727,13333,13939,14545,15152,15758,16364,16970,17576,18182,18788,19394,20000,20606,21212,21818,22424,23030,23636,24242,24848,25455,26061,26667,27273,27879,28485,29091,29697,29697,29091,28485,27879,27273,26667,26061,25455,24848,24242,23636,23030,22424,21818,21212,20606,20000,19394,18788,18182,17576,16970,16364,15758,15152,14545,13939,13333,12727,12121,11515,10909,10303,9697,9091,8485,7879,7273,6667,6061,5455,4848,4242,3636,3030,2424,1818,1212,606,0]
i = 0
# while True:
#     if i == 100:
#         i=0
#     controller_u_output = triangle_angle[i]
#     ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
#     i = i+1
#     time.sleep(1/10)

ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))