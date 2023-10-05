import numpy as np
e = 1
beta_r = 10         # 10
m_0 = 1            # 1
f_2_bar = 1.5       # 1.5
eta = 1          # 0.01
rho = 1.5           # 1 error限制範圍
u_l2 = (m_0/f_2_bar)*((eta*np.log(np.cosh(rho))*np.tanh(e))/((np.log(np.cosh(rho))-np.log(np.cosh(e)))**2))
print(u_l2)