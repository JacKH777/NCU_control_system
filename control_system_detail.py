import numpy as np
from scipy.interpolate import interp1d

class Control:
    def __init__(self):
        self.e = 0
        self.dot_e = 0

    def insert(self, e):
        self.dot_e = (e - self.e)/0.1
        self.e = e

    def get_e(self):
        return self.e

    def get_dot_e(self):
        return self.dot_e

def get_x(angle):
    # mm
    L0 = 300
    L1 = 50
    L2 = 250
    #

    angle = 180 - angle
    L = np.sqrt(L1**2 + L2**2 - 2*L1*L2*np.cos(angle*np.pi/180))
    x = L0 - L
    return x

def control_system(desire_angle,actual_angle,learning_array,array_index,first_period, C):

    e = (get_x(actual_angle) - get_x(desire_angle))
    C.insert(e)

    ################## Control Value ##########################
    smc_lambda = 0.2 # 越快到滑膜面
    k_l1 = 0.5
    k_l2 = 0.01 # 趨近速度增加，抖振增加
    beta_r = 10
    m_0 = 1
    f_2_bar = 1.5
    eta = 0.01

    rho = 10 # error限制範圍
    ###########################################################

    s = smc_lambda * C.get_e() + C.get_dot_e()

    ################### Repetive Learning #####################
    if learning_array[array_index] > beta_r:
        learning_array[array_index] = beta_r
    elif learning_array[array_index] < -beta_r:
        learning_array[array_index] = -beta_r

    if first_period==True:
        w_r_head = k_l1*s*(array_index/200)
        learning_array[array_index] = w_r_head
        if array_index == 200:
            first_period = False
    else:
        w_r_head = learning_array[array_index] + k_l1*s
        learning_array[array_index] = w_r_head
    #########################################################

    u = (m_0/f_2_bar)*((-k_l2 * np.sign(s)) - np.tanh(e) - w_r_head - (eta*np.log(np.cosh(rho))*np.tanh(e))/((np.log(np.cosh(rho))-np.log(np.cosh(e)))**2))

    return u, learning_array, first_period, C

def return_simulation_pma_angle(df_pma_angle,voltage_65535):
    interpolated_function = interp1d(df_pma_angle[1], df_pma_angle[2], kind='linear', fill_value='extrapolate')
    pma_angle = interpolated_function(voltage_65535)
    
    return pma_angle 

if __name__ == '__main__':
    e = (get_x(20) - get_x(25))
    print(e)
 