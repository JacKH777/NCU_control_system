import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
import time


class fuzzy_system:
    def __init__(self):
        # fuzzy rule
        self.error_gauss_center = [-5, -3, -1, 0, 1, 3, 5]
        self.delta_gauss_center = [-5, -3, -1, 0, 1, 3, 5]
        self.output_gauss_center = [-5, -3, -1, 0, 1, 3, 5]

        self.error_gauss_width = [5, 5, 5, 5, 5, 5, 5]
        self.delta_gauss_width = [5, 5, 5, 5, 5, 5, 5]
        self.output_gauss_width = [5, 5, 5, 5, 5, 5, 5]

    def get_gauss_center(self, case, index):
        if case == 'error':
            return self.error_gauss_center[index]
        elif case == 'delta':
            return self.delta_gauss_center[index]
        elif case == 'output':
            return self.output_gauss_center[index]
        
    def get_gauss_width(self, case, index):
        if case == 'error':
            return self.error_gauss_width[index]
        elif case == 'delta':
            return self.delta_gauss_width[index]
        elif case == 'output':
            return self.output_gauss_width[index]
    
    def restart_system(self):

        input_error = np.arange(-26, 26, 0.1, np.float32)
        input_delta = np.arange(-26, 26, 0.1, np.float32)
        output_u = np.arange(-26, 26, 0.1, np.float32)

        error = ctrl.Antecedent(input_error, 'error')
        delta = ctrl.Antecedent(input_delta, 'delta')
        self.output = ctrl.Consequent(output_u, 'output')

        error['nb'] = fuzz.gaussmf(error.universe,self.error_gauss_center[0], self.error_gauss_width[0])
        error['nm'] = fuzz.gaussmf(error.universe, self.error_gauss_center[1], self.error_gauss_width[1])
        error['ns'] = fuzz.gaussmf(error.universe, self.error_gauss_center[2], self.error_gauss_width[2])
        error['zo'] = fuzz.gaussmf(error.universe, self.error_gauss_center[3], self.error_gauss_width[3])
        error['ps'] = fuzz.gaussmf(error.universe, self.error_gauss_center[4], self.error_gauss_width[4])
        error['pm'] = fuzz.gaussmf(error.universe, self.error_gauss_center[5], self.error_gauss_width[5])
        error['pb'] = fuzz.gaussmf(error.universe, self.error_gauss_center[6], self.error_gauss_width[6])

        delta['nb'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[0], self.delta_gauss_width[0])
        delta['nm'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[1], self.delta_gauss_width[1])
        delta['ns'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[2], self.delta_gauss_width[2])
        delta['zo'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[3], self.delta_gauss_width[3])
        delta['ps'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[4], self.delta_gauss_width[4])
        delta['pm'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[5], self.delta_gauss_width[5])
        delta['pb'] = fuzz.gaussmf(delta.universe, self.delta_gauss_center[6], self.delta_gauss_width[6])

        self.output['nb'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[0], self.output_gauss_width[0])
        self.output['nm'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[1], self.output_gauss_width[1])
        self.output['ns'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[2], self.output_gauss_width[2])
        self.output['zo'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[3], self.output_gauss_width[3])
        self.output['ps'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[4], self.output_gauss_width[4])
        self.output['pm'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[5], self.output_gauss_width[5])
        self.output['pb'] = fuzz.gaussmf(self.output.universe, self.output_gauss_center[6], self.output_gauss_width[6])

        rule0 = ctrl.Rule(antecedent=(error['nb'] & delta['nb']), consequent=self.output['nb'], label='rule nb')

        rule1 = ctrl.Rule(antecedent=((error['nm'] & delta['nb']) |
                                    (error['ns'] & delta['nb']) |
                                    (error['nb'] & delta['nm']) |
                                    (error['nm'] & delta['nm']) |
                                    (error['nb'] & delta['ns'])
                                    ), consequent=self.output['nm'], label='rule nm')

        rule2 = ctrl.Rule(antecedent=((error['zo'] & delta['nb']) |
                                    (error['ps'] & delta['nb']) |
                                    (error['pm'] & delta['nb']) |
                                    (error['ns'] & delta['nm']) |
                                    (error['zo'] & delta['nm']) |
                                    (error['ps'] & delta['nm']) |
                                    (error['nm'] & delta['ns']) |
                                    (error['ns'] & delta['ns']) |
                                    (error['zo'] & delta['ns']) |
                                    (error['nb'] & delta['zo']) |
                                    (error['nm'] & delta['zo']) |
                                    (error['ns'] & delta['zo']) |
                                    (error['nb'] & delta['ps']) |
                                    (error['nm'] & delta['ps']) |
                                    (error['nb'] & delta['pm']) 
                                    ), consequent=self.output['ns'], label='rule ns')

        rule3 = ctrl.Rule(antecedent=((error['pb'] & delta['nb']) |
                                    (error['pm'] & delta['nm']) |
                                    (error['ps'] & delta['ns']) |
                                    (error['zo'] & delta['zo']) |
                                    (error['ns'] & delta['ps']) |
                                    (error['nm'] & delta['pm']) |
                                    (error['nb'] & delta['pb'])
                                    ), consequent=self.output['zo'], label='rule zo')

        rule4 = ctrl.Rule(antecedent=((error['pb'] & delta['nm']) |
                                    (error['pm'] & delta['ns']) |
                                    (error['pb'] & delta['ns']) |
                                    (error['ps'] & delta['zo']) |
                                    (error['pm'] & delta['zo']) |
                                    (error['pb'] & delta['zo']) |
                                    (error['pm'] & delta['ps']) |
                                    (error['ps'] & delta['ps']) |
                                    (error['zo'] & delta['ps']) |
                                    (error['ps'] & delta['pm']) |
                                    (error['zo'] & delta['pm']) |
                                    (error['ns'] & delta['pm']) |
                                    (error['ns'] & delta['pb']) |
                                    (error['nm'] & delta['pb']) |
                                    (error['zo'] & delta['pb']) 
                                    ), consequent=self.output['ps'], label='rule ps')

        rule5 = ctrl.Rule(antecedent=((error['pb'] & delta['ps']) |
                                    (error['pm'] & delta['pm']) |
                                    (error['pb'] & delta['pm']) |
                                    (error['ps'] & delta['pb']) |
                                    (error['pm'] & delta['pb'])
                                    ), consequent=self.output['pm'], label='rule pm')

        rule6 = ctrl.Rule(antecedent=(error['pb'] & delta['pb']), consequent=self.output['pb'], label='rule pb')

        system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4, rule5, rule6])
        self.sim = ctrl.ControlSystemSimulation(system)
        

    def calculate(self,error,delta):
        self.sim.input['error'] = error
        self.sim.input['delta'] = delta
        self.sim.compute()
        # self.output.view(sim=self.sim)
        output_m = self.output.Output_membership(sim=self.sim)
        new_u = self.sim.output['output']
        return output_m, new_u 
    
    def output_gauss_learning(self,error,delta,output_learning_rate,error_learning_rate,delta_learning_rate,output_array_m):
        output_array_m = np.asarray(output_array_m)
        output_g_center = np.asarray(self.output_gauss_center)
        output_g_width = np.asarray(self.output_gauss_width)
        sum_output = np.sum(output_array_m * output_g_center)
        new_output_gauss_center = output_g_center + (output_learning_rate * error * output_g_width * output_array_m / sum_output)
        fun = output_g_center/ (sum_output**2)*(output_g_width * sum_output) - np.sum(output_g_width * output_g_center)
        new_output_gauss_width = output_g_width + (output_learning_rate * error * output_array_m  * fun)

        # center
        error_g_center = np.asarray(self.error_gauss_center)
        error_g_width = np.asarray(self.error_gauss_width)
        error_memberships = np.asarray([fuzz.gaussmf(error, c, s) for c, s in zip(error_g_center, error_g_width)])

        delta_g_center = np.asarray(self.delta_gauss_center)
        delta_g_width = np.asarray(self.delta_gauss_width)
        delta_memberships = np.asarray([fuzz.gaussmf(delta, c, s) for c, s in zip(delta_g_center, delta_g_width)])

        smaller = np.less(error_memberships, delta_memberships)  # 或者使用 arr1 < arr2
        error_used = np.where(smaller, 1, 0)
        delta_used = 1-error_used

        new_error_gauss_center = error_g_center - (error_learning_rate*2*(error - error_g_center)/(error_g_width**2)*(-error*output_g_width*fun*error_used))
        new_error_gauss_width = error_g_center - (error_learning_rate*2*(error - error_g_center)**2/(error_g_width**3)*(-error*output_g_width*fun*error_used))

        new_delta_gauss_center = delta_g_center - (delta_learning_rate*2*(delta - delta_g_center)/(delta_g_width**2)*(-error*output_g_width*fun*delta_used))
        new_delta_gauss_width = delta_g_center - (delta_learning_rate*2*(delta - delta_g_center)**2/(delta_g_width**3)*(-error*output_g_width*fun*delta_used))
        return new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width

 

    def new_output_gauss(self,new_output_gauss_center,new_output_gauss_width,new_error_gauss_center,new_error_gauss_width,new_delta_gauss_center,new_delta_gauss_width):
        self.output_gauss_center = new_output_gauss_center
        self.output_gauss_width = new_output_gauss_width
        self.error_gauss_center = new_error_gauss_center
        self.error_gauss_width = new_error_gauss_width
        self.delta_gauss_center = new_delta_gauss_center
        self.error_gauss_width = new_delta_gauss_width
        
class RealTimeGaussianPlot:
    def __init__(self, num_lines=7, x_range=(-13, 13), y_range=(0, 1.2), line_colors=None):
        self.num_lines = num_lines
        self.x = np.linspace(x_range[0], x_range[1], 400)
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(y_range[0], y_range[1])
        self.lines = [self.ax.plot(self.x, np.zeros_like(self.x), color=line_colors[i] if line_colors else None)[0] for i in range(num_lines)]


    def gaussian(self, x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma)**2)

    def set_data(self, centers = [-15, -10, -5, 0, 5, 10, 15], widths = [5, 5, 5, 5, 5, 5, 5]):
        # self.new_centers = centers
        # self.new_widths = widths
        self.update(centers, widths)
        plt.draw()
        plt.pause(0.1)

    # def animate_update(self, frame):
    #     if hasattr(self, 'new_centers') and hasattr(self, 'new_widths'):
    #         self.update(self.new_centers, self.new_widths)

    def update(self, centers, widths):
        if len(centers) != self.num_lines or len(widths) != self.num_lines:
            raise ValueError("Length of centers and widths must match num_lines")

        for line, mean, sigma in zip(self.lines, centers, widths):
            y = self.gaussian(self.x, mean, sigma)
            line.set_ydata(y)
        # self.ax.relim()  # 重新计算轴的界限
        # self.ax.autoscale_view()  # 自动调整轴界限

class self_fuzzy_system:
    def __init__(self):
        # membership default
        self.error_gauss_center = np.asarray([-5, -3, -1, 0, 1, 3, 5])
        self.delta_gauss_center = np.asarray([-5, -3, -1, 0, 1, 3, 5])
        self.output_gauss_center = np.asarray([-10, -5, -3, 0, 3, 5, 10])

        error_width = 0.6
        delta_width = 0.6
        output_width = 0.6
        # self.error_gauss_width = np.asarray([error_width, error_width, error_width, error_width, error_width, error_width, error_width])
        # self.delta_gauss_width = np.asarray([delta_width, delta_width, delta_width, delta_width, delta_width, delta_width, delta_width])
        # self.output_gauss_width = np.asarray([output_width, output_width, output_width, output_width, output_width, output_width, output_width])
        
        self.error_gauss_width = np.asarray([1.5, 1, 1, 1, 1, 1, 1.5])
        self.delta_gauss_width = np.asarray([1.5, 1, 1, 1, 1, 1, 1.5])
        self.output_gauss_width = np.asarray([1.5, 1, 1, 1, 1, 1, 1.5])
        self.error_gauss_member = np.zeros(7)
        self.delta_gauss_member = np.zeros(7)
        self.output_gauss_member = np.zeros(7)

        self.vectorized_gaussian = np.vectorize(self.gaussian)

        self.rule_table = np.zeros((7, 7))
    
    def gaussian(self, x, center, width):
        return np.exp(-0.5 * ((x - center) / width)**2)
    
    def calcualte_rule_table(self):
        self.error_count = np.zeros(7)
        self.delta_count = np.zeros(7)
        self.error_gauss_member[self.error_gauss_member<0.01] = 0
        self.delta_gauss_member[self.delta_gauss_member<0.01] = 0
        for i, error_value in enumerate(self.error_gauss_member):
            for j, delta_value in enumerate(self.delta_gauss_member):
                if error_value < delta_value:
                    self.rule_table[i, j] = error_value
                    self.error_count[i] = self.error_count[i] + 1
                elif error_value == delta_value and error_value == 0:
                    self.rule_table[i, j] = 0
                else:
                    self.rule_table[i, j] = delta_value
                    self.delta_count[j] = self.delta_count[j] + 1 
 

    def get_gauss_member(self):
        rule_0 = [self.rule_table[0, 0],self.rule_table[0, 1],self.rule_table[1, 0]]
        self.output_gauss_member[0] = min(np.sum(rule_0), 1)

        rule_1 = [  self.rule_table[0, 2],self.rule_table[0, 3],self.rule_table[0, 4],
                    self.rule_table[1, 1],self.rule_table[1, 2],self.rule_table[1, 3],
                    self.rule_table[2, 0],self.rule_table[2, 1],self.rule_table[2, 2],
                    self.rule_table[3, 0],self.rule_table[3, 1],
                    self.rule_table[4, 0]]
        # filtered_rule_1 = [v for v in rule_1 if v != 0]
        # if filtered_rule_1:
        #     self.output_gauss_member[1] = np.min(filtered_rule_1)
        # else:
        #     self.output_gauss_member[1] = 0
        # self.output_gauss_member[1] = np.min(rule_1)
        self.output_gauss_member[1] = min(np.sum(rule_1), 1)

        rule_2 = [  self.rule_table[0, 5],
                    self.rule_table[1, 4],
                    self.rule_table[2, 3],
                    self.rule_table[3, 2],
                    self.rule_table[4, 1],
                    self.rule_table[5, 0]]
        # filtered_rule_2 = [v for v in rule_2 if v != 0]
        # if filtered_rule_2:
        #     self.output_gauss_member[2] = np.min(filtered_rule_2)
        # else:
        #     self.output_gauss_member[2] = 0
        self.output_gauss_member[2] = min(np.sum(rule_2), 1)

        rule_3 = [  self.rule_table[0, 6],
                    self.rule_table[1, 5],
                    self.rule_table[2, 4],
                    self.rule_table[3, 3],
                    self.rule_table[4, 2],
                    self.rule_table[5, 1],
                    self.rule_table[6, 0]]
        # filtered_rule_3 = [v for v in rule_3 if v != 0]
        # if filtered_rule_3:
        #     self.output_gauss_member[3] = np.min(filtered_rule_3)
        # else:
        #     self.output_gauss_member[3] = 0
        self.output_gauss_member[3] = min(np.sum(rule_3), 1)

        rule_4 = [  self.rule_table[1, 6],
                    self.rule_table[2, 5],
                    self.rule_table[3, 4],
                    self.rule_table[4, 3],
                    self.rule_table[5, 2],
                    self.rule_table[6, 1]]
        # filtered_rule_4 = [v for v in rule_4 if v != 0]
        # if filtered_rule_4:
        #     self.output_gauss_member[4] = np.min(filtered_rule_4)
        # else:
        #     self.output_gauss_member[4] = 0
        self.output_gauss_member[4] = min(np.sum(rule_4), 1)

        rule_5 = [  self.rule_table[2, 6],
                    self.rule_table[3, 6],self.rule_table[3, 5],
                    self.rule_table[4, 6],self.rule_table[4, 5],self.rule_table[4, 4],
                    self.rule_table[5, 5],self.rule_table[5, 4],self.rule_table[5, 3],
                    self.rule_table[6, 4],self.rule_table[6, 3],self.rule_table[6, 2]]
        # filtered_rule_5 = [v for v in rule_5 if v != 0]
        # if filtered_rule_5:
        #     self.output_gauss_member[5] = np.min(filtered_rule_5)
        # else:
        #     self.output_gauss_member[5] = 0
        self.output_gauss_member[5] = min(np.sum(rule_5), 1)

        rule_6 = [self.rule_table[6, 5],self.rule_table[5, 6],self.rule_table[6, 6]]
        self.output_gauss_member[6] = min(np.sum(rule_6), 1)


    def defuzz_output_gauss(self,error):
        # if np.all(self.output_gauss_member == 0) and error>0:
        #     self.output_gauss_member[6]=1
        #     self.output_gauss_member[5]=1
        # elif np.all(self.output_gauss_member == 0) and error<0:
        #     self.output_gauss_member[0]=1
        #     self.output_gauss_member[1]=1
        # print('error:',self.error_gauss_member)
        # print('rror:',self.output_gauss_member)
        control_u = np.sum(self.output_gauss_center * self.output_gauss_width * self.output_gauss_member)/np.sum(self.output_gauss_width * self.output_gauss_member)
        return control_u
    
    # def output_gauss_learning(self,error,delta,output_learning_rate = 0,error_learning_rate = 0,delta_learning_rate = 0):
        
    #     # epsilon = 1e-10  # 避免除以零
    #     sum_output = np.sum(self.output_gauss_member * self.output_gauss_width)
    #     new_output_gauss_center = self.output_gauss_center + (output_learning_rate * error * self.output_gauss_width * self.output_gauss_member / sum_output)
    #     fun = (self.output_gauss_center*((self.output_gauss_center* sum_output) - np.sum(self.output_gauss_center * self.output_gauss_member*self.output_gauss_width)))/(sum_output**2)
    #     new_output_gauss_width = self.output_gauss_width + (output_learning_rate * error * self.output_gauss_member  * fun)

    #     # # center
    #     # smaller = np.less(self.error_gauss_member, self.delta_gauss_member)  # 或者使用 arr1 < arr2
    #     # error_used = np.where(smaller, 1, 0)
    #     # delta_used = 1-error_used

    #     new_error_gauss_center = self.error_gauss_center - (((error_learning_rate*2*(error - self.error_gauss_center))/(self.error_gauss_width**2))*((-error)*self.output_gauss_width*fun*self.error_count))
    #     new_error_gauss_width = self.error_gauss_width - (error_learning_rate*2*((error - self.error_gauss_center)**2)/(self.error_gauss_width**3)*((-error)*self.output_gauss_width*fun*self.error_count))

    #     new_delta_gauss_center = self.delta_gauss_center - (((delta_learning_rate*2*(delta - self.delta_gauss_center))/(self.delta_gauss_width**2))*((-error)*self.output_gauss_width*fun*self.delta_count))
    #     new_delta_gauss_width = self.delta_gauss_width - (delta_learning_rate*2*((delta - self.delta_gauss_center)**2)/(self.delta_gauss_width**3)*((-error)*self.output_gauss_width*fun*self.delta_count))
        
    #     # 更新
    #     self.output_gauss_center = new_output_gauss_center
    #     self.output_gauss_width = new_output_gauss_width
    #     self.error_gauss_center = new_error_gauss_center
    #     self.error_gauss_width = new_error_gauss_width
    #     self.delta_gauss_center = new_delta_gauss_center
    #     self.delta_gauss_width = new_delta_gauss_width

    def fuzzy_rule(self,error,delta): 
        if error < self.error_gauss_center[0]:
            error = self.error_gauss_center[0]
        elif error >self.error_gauss_center[6]:
            error = self.error_gauss_center[6]
        if delta < self.delta_gauss_center[0]:
            delta = self.delta_gauss_center[0]
        elif delta >self.delta_gauss_center[6]:
            delta = self.delta_gauss_center[6]
        self.error_gauss_member = self.vectorized_gaussian(error, self.error_gauss_center, self.error_gauss_width)
        self.delta_gauss_member = self.vectorized_gaussian(delta, self.delta_gauss_center, self.delta_gauss_width)
        # self.error_gauss_member[0] =  1 / (1 + np.exp(self.error_gauss_width[0]*(error - self.error_gauss_center[0])))
        # self.delta_gauss_member[0] =  1 / (1 + np.exp(self.delta_gauss_width[0]*(delta - self.delta_gauss_center[0])))
        # self.error_gauss_member[6] =  1 / (1 + np.exp(self.error_gauss_width[6]*(error - self.error_gauss_center[6])))
        # self.delta_gauss_member[6] =  1 / (1 + np.exp(self.delta_gauss_width[6]*(delta - self.delta_gauss_center[6])))
        self.calcualte_rule_table()
        self.get_gauss_member()
        u = self.defuzz_output_gauss(error)
        return u,self.error_gauss_center,self.error_gauss_width



