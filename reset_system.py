import serial
import time

ser_1 = serial.Serial("COM6", 115200)

controller_u_output = 0
triangle_angle = [0,120,240,361,481,601,721,842,962,1082,1202,1323,1443,1563,1683,1804,1924,2044,2164,2285,2405,2525,2645,2766,2886,3006,3126,3246,3367,3487,3607,3727,3848,3968,4088,4208,4329,4449,4569,4689,4810,4930,5050,5170,5291,5411,5531,5651,5772,5892,6012,6132,6253,6373,6493,6613,6733,6854,6974,7094,7214,7335,7455,7575,7695,7816,7936,8056,8176,8297,8417,8537,8657,8778,8898,9018,9138,9259,9379,9499,9619,9739,9860,9980,10100,10220,10341,10461,10581,10701,10822,10942,11062,11182,11303,11423,11543,11663,11784,11904,12024,12144,12265,12385,12505,12625,12745,12866,12986,13106,13226,13347,13467,13587,13707,13828,13948,14068,14188,14309,14429,14549,14669,14790,14910,15030,15150,15271,15391,15511,15631,15752,15872,15992,16112,16232,16353,16473,16593,16713,16834,16954,17074,17194,17315,17435,17555,17675,17796,17916,18036,18156,18277,18397,18517,18637,18758,18878,18998,19118,19238,19359,19479,19599,19719,19840,19960,20080,20200,20321,20441,20561,20681,20802,20922,21042,21162,21283,21403,21523,21643,21764,21884,22004,22124,22244,22365,22485,22605,22725,22846,22966,23086,23206,23327,23447,23567,23687,23808,23928,24048,24168,24289,24409,24529,24649,24770,24890,25010,25130,25251,25371,25491,25611,25731,25852,25972,26092,26212,26333,26453,26573,26693,26814,26934,27054,27174,27295,27415,27535,27655,27776,27896,28016,28136,28257,28377,28497,28617,28737,28858,28978,29098,29218,29339,29459,29579,29699,29820,29940,29940,29820,29699,29579,29459,29339,29218,29098,28978,28858,28737,28617,28497,28377,28257,28136,28016,27896,27776,27655,27535,27415,27295,27174,27054,26934,26814,26693,26573,26453,26333,26212,26092,25972,25852,25731,25611,25491,25371,25251,25130,25010,24890,24770,24649,24529,24409,24289,24168,24048,23928,23808,23687,23567,23447,23327,23206,23086,22966,22846,22725,22605,22485,22365,22244,22124,22004,21884,21764,21643,21523,21403,21283,21162,21042,20922,20802,20681,20561,20441,20321,20200,20080,19960,19840,19719,19599,19479,19359,19238,19118,18998,18878,18758,18637,18517,18397,18277,18156,18036,17916,17796,17675,17555,17435,17315,17194,17074,16954,16834,16713,16593,16473,16353,16232,16112,15992,15872,15752,15631,15511,15391,15271,15150,15030,14910,14790,14669,14549,14429,14309,14188,14068,13948,13828,13707,13587,13467,13347,13226,13106,12986,12866,12745,12625,12505,12385,12265,12144,12024,11904,11784,11663,11543,11423,11303,11182,11062,10942,10822,10701,10581,10461,10341,10220,10100,9980,9860,9739,9619,9499,9379,9259,9138,9018,8898,8778,8657,8537,8417,8297,8176,8056,7936,7816,7695,7575,7455,7335,7214,7094,6974,6854,6733,6613,6493,6373,6253,6132,6012,5892,5772,5651,5531,5411,5291,5170,5050,4930,4810,4689,4569,4449,4329,4208,4088,3968,3848,3727,3607,3487,3367,3246,3126,3006,2886,2766,2645,2525,2405,2285,2164,2044,1924,1804,1683,1563,1443,1323,1202,1082,962,842,721,601,481,361,240,120,0]
triangle_angle_0_255 = [0,1,3,4,5,6,8,9,10,12,13,14,15,17,18,19,21,22,23,24,26,27,28,30,31,32,33,35,36,37,39,40,41,42,44,45,46,48,49,50,51,53,54,55,57,58,59,60,62,63,64,66,67,68,69,71,72,73,75,76,77,78,80,81,82,84,85,86,87,89,90,91,93,94,95,96,98,99,100,102,103,104,105,107,108,109,111,112,113,114,116,117,118,120,121,122,123,125,126,127,127,126,125,123,122,121,120,118,117,116,114,113,112,111,109,108,107,105,104,103,102,100,99,98,96,95,94,93,91,90,89,87,86,85,84,82,81,80,78,77,76,75,73,72,71,69,68,67,66,64,63,62,60,59,58,57,55,54,53,51,50,49,48,46,45,44,42,41,40,39,37,36,35,33,32,31,30,28,27,26,24,23,22,21,19,18,17,15,14,13,12,10,9,8,6,5,4,3,1,0]
i = 0
first_count = 0
actual_angle = 25
test = 0
# while True:
#     if test < 200:
#         i=0
#     if i == 200:
#         i=0
#     controller_u_output = triangle_angle[0]
#     ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))

#     i = i+1
#     time.sleep(0.1)

ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))