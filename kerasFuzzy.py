import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from encoder_function import encoder
import time
# tf.keras.backend.clear_session()
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '-1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ANFIS:

    def __init__(self,learning_rate=0.01):

        self.mu_error = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_error = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.mu_delta = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_delta = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([1, 1, 1, 1, 1, 1, 1]), dtype=tf.float64)
        self.trainable_variables = [self.mu_error, self.sigma_error,self.mu_delta, self.sigma_delta, self.y]


        initial_learning_rate = 0.4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,
            decay_rate=0.95,
            staircase=True)
        # self.optimizer = tf.optimizers.Adam(learning_rate = learning_rate) # Optimization step
        # self.optimizer = tf.optimizers.RMSprop(learning_rate=0.01)
        # self.optimizer = tf.optimizers.Adadelta(learning_rate=1, rho=0.9)
        # # self.optimizer = tf.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
        # self.optimizer = tf.optimizers.Adam(learning_rate = 0) 
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.6)
        # self.optimizer = tf.optimizers.Adagrad(learning_rate=0.01)


    def train(self, error,delta, targets, actual_angle):
        with tf.GradientTape() as tape:
            self.error = tf.convert_to_tensor( error, dtype=tf.float64)
            self.delta = tf.convert_to_tensor( delta, dtype=tf.float64)
            targets = tf.convert_to_tensor(targets, dtype=tf.float64)
            actual_angle = tf.convert_to_tensor(actual_angle, dtype=tf.float64)
            # self.error =error
            # self.delta = delta
            # targets = targets
            # actual_angle = actual_angle

            self.error_first = 1 / (1 + tf.exp(self.sigma_error[0]*(self.error[0] - self.mu_error[0])))
            self.delta_first = 1 / (1 + tf.exp(self.sigma_delta[0]*(self.delta[0]  - self.mu_delta[0])))
            self.error_last = 1 / (1 + tf.exp(-self.sigma_error[-1]*(self.error[0]  - self.mu_error[-1])))
            self.delta_last = 1 / (1 + tf.exp(-self.sigma_delta[-1]*(self.delta[0]  - self.mu_delta[-1])))
            # self.rul_error = tf.reduce_prod(
            # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6])),
            #            (5, 1)), axis=1)  # Rule activations
            self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6]))
            # self.rul_delta = tf.reduce_prod(
            # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6])),
            #            (5, 1)), axis=1)  # Rule activations
            
            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.mf_error = tf.concat([self.error_first, self.rul_error, self.error_last], axis=0)
            self.mf_delta = tf.concat([self.delta_first, self.rul_delta, self.delta_last], axis=0)
            # print('error:',self.mf_error )
            # print('delta:',self.mf_delta )
            # rul_error_expanded = tf.expand_dims(self.mf_error, 1)  # 扩展为 [ 7, 1] 然后复制
            # rul_delta_expanded = tf.expand_dims(self.mf_delta, 0)  # 扩展为 [ 1, 7] 然后复制

            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.mf_error, axis=1), tf.expand_dims(self.mf_delta, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out
            self.out = actual_angle + (0.001*self.out)
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            loss = mse_loss(targets, self.out)
        if tf.math.is_nan(self.u):
            self.u = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return tf.squeeze(self.u)

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # name = name + '.npy'
        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            self.mu_error.numpy(),
            self.sigma_error.numpy(),
            self.mu_delta.numpy(),
            self.sigma_delta.numpy(),
            self.y.numpy(),
            self.y_sigma.numpy()
        ))
        return all_data

    def predict(self, error, delta,):
        self.error = tf.convert_to_tensor( error, dtype=tf.float64)
        self.delta = tf.convert_to_tensor( delta, dtype=tf.float64)

        self.error_first = 1 / (1 + tf.exp(self.sigma_error[0]*(self.error[0] - self.mu_error[0])))
        self.delta_first = 1 / (1 + tf.exp(self.sigma_delta[0]*(self.delta[0]  - self.mu_delta[0])))
        self.error_last = 1 / (1 + tf.exp(-self.sigma_error[-1]*(self.error[0]  - self.mu_error[-1])))
        self.delta_last = 1 / (1 + tf.exp(-self.sigma_delta[-1]*(self.delta[0]  - self.mu_delta[-1])))
        # self.rul_error = tf.reduce_prod(
        # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6])),
        #            (5, 1)), axis=1)  # Rule activations
        self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6]))
        self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6]))
        # self.rul_delta = tf.reduce_prod(
        # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6])),
        #            (5, 1)), axis=1)  # Rule activations
        
        self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
        self.error_last = tf.expand_dims(self.error_last, -1)
        self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
        self.error_last = tf.expand_dims(self.error_last, -1)
        self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
        self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
        self.delta_last = tf.expand_dims(self.delta_last, -1)
        self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
        self.delta_last = tf.expand_dims(self.delta_last, -1)
        self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


        self.mf_error = tf.concat([self.error_first, self.rul_error, self.error_last], axis=0)
        self.mf_delta = tf.concat([self.delta_first, self.rul_delta, self.delta_last], axis=0)

        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(self.mf_error, axis=1), tf.expand_dims(self.mf_delta, axis=0))

        rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        self.rul = tf.reshape(self.rul, (1, 7))
        # Fuzzy base expansion function:
        self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
        den = tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
        self.out = tf.divide(num, den)
        self.u = self.out

        return tf.squeeze(self.u)

    def load_model(self, path):
        # 加载数据
        all_data_loaded = np.load(path)
        # print(all_data_loaded[-1])
        # 从后面取出最后 42 个数据
        all_data_loaded = all_data_loaded[-42:]
        
        # 检查形状是否正确
        if all_data_loaded.shape[0] != 42:
            raise ValueError("Expected 42 data points, but got {}".format(all_data_loaded.shape[0]))
        
        # 重塑为 6 列，7 行
        all_data_loaded = all_data_loaded.reshape(7, 6)
        
        # 分别赋值给不同的变量
        self.mu_error.assign(all_data_loaded[:, 0])        # 取第 1 列
        self.sigma_error.assign(all_data_loaded[:, 1])     # 取第 2 列
        self.mu_delta.assign(all_data_loaded[:, 2])        # 取第 3 列
        self.sigma_delta.assign(all_data_loaded[:, 3])     # 取第 4 列
        self.y.assign(all_data_loaded[:, 4])               # 取第 5 列
        self.y_sigma.assign(all_data_loaded[:, 5])         # 取第 6 列

class ori_ANFIS:

    def __init__(self,learning_rate=0.005):

        self.mu_error = tf.Variable(np.asarray([-5, -3, -1, 0, 1, 3, 5]), dtype=tf.float32)
        self.sigma_error = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
        self.mu_delta = tf.Variable(np.asarray([-5, -3, -1, 0, 1, 3, 5]), dtype=tf.float32)
        self.sigma_delta = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
        self.y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float32)
        self.y_sigma = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
  

    def predict(self, error,delta):
        with tf.GradientTape() as tape:
            self.error = tf.convert_to_tensor( error, dtype=tf.float32)
            self.delta = tf.convert_to_tensor( delta, dtype=tf.float32)

            self.error_first = 1 / (1 + tf.exp(self.sigma_error[0]*(self.error[0] - self.mu_error[0])))
            self.delta_first = 1 / (1 + tf.exp(self.sigma_delta[0]*(self.delta[0]  - self.mu_delta[0])))
            self.error_last = 1 / (1 + tf.exp(-self.sigma_error[-1]*(self.error[0]  - self.mu_error[-1])))
            self.delta_last = 1 / (1 + tf.exp(-self.sigma_delta[-1]*(self.delta[0]  - self.mu_delta[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6]))
 
            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)

            self.mf_error = tf.concat([self.error_first, self.rul_error, self.error_last], axis=0)
            self.mf_delta = tf.concat([self.delta_first, self.rul_delta, self.delta_last], axis=0)
            # rul_error_expanded = tf.expand_dims(self.mf_error, 1)  # 扩展为 [ 7, 1] 然后复制
            # rul_delta_expanded = tf.expand_dims(self.mf_delta, 0)  # 扩展为 [ 1, 7] 然后复制

            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.mf_error, axis=1), tf.expand_dims(self.mf_delta, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out

        return tf.squeeze(self.u)
   
def go_to_desire_angle(encoder, stm32, desire_angle, controller_u):
    fuzzy = ANFIS()
    angle = encoder.get_angle()
    error = desire_angle - angle
    error_dot = 0
    last_error = 0
    ku = 0.005
    idx = 0
    angle_his = np.asarray([])
    while idx < 100:
        # fuzzy.train([error],[error_dot], [desire_angle],[angle])
        error_dot = (error -  last_error)
        last_error = error # 更新過去誤差
        delta_u = fuzzy.predict([error],[error_dot])
        delta_u = delta_u * ku
        controller_u = controller_u + delta_u
        controller_u = float(controller_u)
        controller_u_output = controller_u/10*65535
        controller_u_output = int(controller_u_output)
        if controller_u_output < 0:
            controller_u_output = 0
        stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        idx += 1
        angle = encoder.get_angle()
        error = desire_angle - angle
        angle_his =  np.append(angle_his,angle)
        time.sleep(0.01)
    return angle_his,controller_u

class Torque_ANFIS:


    def __init__(self):

        self.mu_error = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.sigma_error = tf.Variable(np.asarray([1.5, 2, 2, 2, 2, 2, 1.5]), dtype=tf.float64)
        self.mu_delta = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.sigma_delta = tf.Variable(np.asarray([1.5, 2, 2, 2, 2, 2, 1.5]), dtype=tf.float64)
        self.y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([1.5, 2, 2, 2, 2, 2, 1.5]), dtype=tf.float64)
        self.trainable_variables = [self.mu_error, self.sigma_error,self.mu_delta, self.sigma_delta, self.y,self.y_sigma]

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.8)


    def train(self, error,delta, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.error = tf.convert_to_tensor( error, dtype=tf.float64)
            self.delta = tf.convert_to_tensor( delta, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            self.error = self.error * 30
            self.delta = self.delta * 30
            targets_torque = targets_torque * 30
            actual_torque = actual_torque * 30

            self.error_first = 1 / (1 + tf.exp(self.sigma_error[0]*(self.error[0] - self.mu_error[0])))
            self.delta_first = 1 / (1 + tf.exp(self.sigma_delta[0]*(self.delta[0]  - self.mu_delta[0])))
            self.error_last = 1 / (1 + tf.exp(-self.sigma_error[-1]*(self.error[0]  - self.mu_error[-1])))
            self.delta_last = 1 / (1 + tf.exp(-self.sigma_delta[-1]*(self.delta[0]  - self.mu_delta[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6]))

            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
            self.error_last = tf.expand_dims(self.error_last, -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
            self.delta_last = tf.expand_dims(self.delta_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.mf_error = tf.concat([self.error_first, self.rul_error, self.error_last], axis=0)
            self.mf_delta = tf.concat([self.delta_first, self.rul_delta, self.delta_last], axis=0)

            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.mf_error, axis=1), tf.expand_dims(self.mf_delta, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out
            self.out = actual_torque + (0.001*self.out)
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            loss = mse_loss(targets_torque, self.out)
        if tf.math.is_nan(self.u):
            self.u = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return tf.squeeze(self.u)

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # name = name + '.npy'
        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            self.mu_error.numpy(),
            self.sigma_error.numpy(),
            self.mu_delta.numpy(),
            self.sigma_delta.numpy(),
            self.y.numpy(),
            self.y_sigma.numpy()
        ))
        return all_data

    def predict(self, error, delta,):
        self.error = tf.convert_to_tensor( error, dtype=tf.float64)
        self.delta = tf.convert_to_tensor( delta, dtype=tf.float64)
        self.error = self.error * 30
        self.delta = self.delta * 30

        self.error_first = 1 / (1 + tf.exp(self.sigma_error[0]*(self.error[0] - self.mu_error[0])))
        self.delta_first = 1 / (1 + tf.exp(self.sigma_delta[0]*(self.delta[0]  - self.mu_delta[0])))
        self.error_last = 1 / (1 + tf.exp(-self.sigma_error[-1]*(self.error[0]  - self.mu_error[-1])))
        self.delta_last = 1 / (1 + tf.exp(-self.sigma_delta[-1]*(self.delta[0]  - self.mu_delta[-1])))
        # self.rul_error = tf.reduce_prod(
        # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6])),
        #            (5, 1)), axis=1)  # Rule activations
        self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [5]), self.mu_error[1:6])) / tf.square(self.sigma_error[1:6]))
        self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6]))
        # self.rul_delta = tf.reduce_prod(
        # tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [5]), self.mu_delta[1:6])) / tf.square(self.sigma_delta[1:6])),
        #            (5, 1)), axis=1)  # Rule activations
        
        self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
        self.error_last = tf.expand_dims(self.error_last, -1)
        self.error_first = tf.expand_dims(self.error_first, -1)  # 在最后增加一个维度
        self.error_last = tf.expand_dims(self.error_last, -1)
        self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
        self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
        self.delta_last = tf.expand_dims(self.delta_last, -1)
        self.delta_first = tf.expand_dims(self.delta_first, -1)  # 在最后增加一个维度
        self.delta_last = tf.expand_dims(self.delta_last, -1)
        self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


        self.mf_error = tf.concat([self.error_first, self.rul_error, self.error_last], axis=0)
        self.mf_delta = tf.concat([self.delta_first, self.rul_delta, self.delta_last], axis=0)

        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(self.mf_error, axis=1), tf.expand_dims(self.mf_delta, axis=0))

        rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        self.rul = tf.reshape(self.rul, (1, 7))
        # Fuzzy base expansion function:
        self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
        den = tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
        self.out = tf.divide(num, den)
        self.u = self.out

        return tf.squeeze(self.u)

    def load_model(self,path):
        all_data_loaded = np.loadtxt(path, delimiter=',')
    
        # 分列存储到各个变量中，假设我们知道有六列
        self.mu_error.assign(all_data_loaded[:, 0])
        self.sigma_error.assign(all_data_loaded[:, 1])
        self.mu_delta.assign(all_data_loaded[:, 2])
        self.sigma_delta.assign(all_data_loaded[:, 3])
        self.y.assign(all_data_loaded[:, 4])
        self.y_sigma.assign(all_data_loaded[:, 5])

class Torque_ANFIS_1kg:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([0.7, 1.19, 1.68]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([11, 0.5, 11]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([20, 0.05, 0.05, 0.05, 0.05, 0.05, 20]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque, self.mf_y,self.y_sigma]
        initial_learning_rate = 0.4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[2, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 1]+rule_table[1, 0]+rule_table[2, 1]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 2]+rule_table[1, 1]+rule_table[1, 2]+rule_table[2, 2]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[1, 5]+rule_table[2, 4]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[0, 5]+rule_table[1, 6]+rule_table[2, 5]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[0, 6]+rule_table[2, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out
            self.out = actual_torque + (0.001*self.out)
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))
        if tf.math.is_nan(self.u):
            self.u = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
        self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
        self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
        self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))
    
        self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
        self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))


        self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
        self.torque_last = tf.expand_dims(self.torque_last , -1)
        self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
        self.torque_last  = tf.expand_dims(self.torque_last , -1)
        self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
        self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
        self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
        self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
        self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
        self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)

        self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
        self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

        rule_0_sum = rule_table[0, 0] + rule_table[2, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 1]+rule_table[1, 0]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 2]+rule_table[1, 1]+rule_table[1, 2]+rule_table[2, 2]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[1, 5]+rule_table[2, 4]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[0, 5]+rule_table[1, 6]+rule_table[2, 5]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[0, 6]+rule_table[2, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        self.rul = tf.reshape(self.rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
        den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
        self.out = tf.divide(num, den)
        self.u = self.out

        return tf.squeeze(self.u)

    def load_model(self, model_data_flat, max_length = 7):
        # Reshape the flat data back into a 2D array with 6 columns
        model_data = model_data_flat.reshape(-1, 6)
        
        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque = model_data[~np.isnan(model_data[:, 0]), 0]  # Remove NaN from mf_torque
        self.sigma_torque = model_data[~np.isnan(model_data[:, 1]), 1]  # Remove NaN from sigma_torque
        self.mf_delta_torque = model_data[~np.isnan(model_data[:, 2]), 2]  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque = model_data[~np.isnan(model_data[:, 3]), 3]  # Remove NaN from sigma_delta_torque
        self.mf_y = model_data[~np.isnan(model_data[:, 4]), 4]  # Remove NaN from mf_y
        self.y_sigma = model_data[~np.isnan(model_data[:, 5]), 5]  # Remove NaN from y_sigma
        
        print("Model loaded successfully without NaN values!")


class Torque_ANFIS_2kg_multi:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([0.51,0.975, 1.44]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([20, 0.3, 20]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-0.2, -0.15, -0.05, 0, 0.05, 0.15, 0.2]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([150, 0.04, 0.04, 0.02, 0.04, 0.04, 150]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y,self.y_sigma]

        self.mf_torque_predict = tf.Variable(np.asarray([0.51,0.975, 1.44]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([20, 0.3, 20]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-0.2, -0.15, -0.05, 0, 0.05, 0.15, 0.2]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([150, 0.04, 0.04, 0.02, 0.04, 0.04, 150]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)

        initial_learning_rate = 0.002
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.5,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=False,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[2, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 1]+rule_table[2, 1]+rule_table[1, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 2]+rule_table[1, 1]+rule_table[1, 2]+rule_table[2, 2]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[1, 5]+rule_table[2, 4]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[0, 5]+rule_table[2, 5]+rule_table[1, 6]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[0, 6]+rule_table[2, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out

            self.out = actual_torque + (0.001*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        # gradients = [tf.clip_by_value(grad, -0.005, 0.005) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)



    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
    
        rul_error =tf.exp(-tf.square(tf.subtract(torque, self.mf_torque_predict[1])) / tf.square(self.sigma_torque_predict[1]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [5]), self.mf_delta_torque_predict[1:6])) / tf.square(self.sigma_delta_torque_predict[1:6]))


        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[0, 0] + rule_table[2, 0]+rule_table[1, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 1]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 2]+rule_table[1, 1]+rule_table[1, 2]+rule_table[2, 2]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[1, 5]+rule_table[2, 4]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[0, 5]+rule_table[2, 5]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[0, 6]+rule_table[2, 6]+rule_table[1, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(rul, self.mf_y_predict),self.y_sigma_predict), axis=1)
        den =tf.reduce_sum(tf.multiply(rul,self.y_sigma_predict), axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")

class Torque_ANFIS_multi:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.25]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y,self.y_sigma]

        self.mf_torque_predict = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)

        # initial_learning_rate = 0.4
        initial_learning_rate = 0.4
        # initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=True,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[2, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 1]+rule_table[0, 2] + rule_table[1, 2]+rule_table[2, 2]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[1, 5]+rule_table[2, 5]+rule_table[2, 6]+rule_table[0, 6]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[2, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)

            # rule_0_sum = rule_table[2, 0]+rule_table[2, 1]+rule_table[1, 0]
            # rule_0 = tf.minimum(rule_0_sum, 1)
            # rule_1_sum =  rule_table[0, 0] + rule_table[0, 1]+rule_table[1, 1]+rule_table[2, 2]
            # rule_1 = tf.minimum(rule_1_sum, 1)
            # rule_2_sum = rule_table[0, 2]+rule_table[1, 2]
            # rule_2 = tf.minimum(rule_2_sum, 1)
            # rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            # rule_3 = tf.minimum(rule_3_sum, 1)
            # rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            # rule_4 = tf.minimum(rule_4_sum, 1)
            # rule_5_sum = rule_table[0, 5]+rule_table[0, 6]+rule_table[2, 6]+rule_table[2, 4]
            # rule_5 = tf.minimum(rule_5_sum, 1)            
            # rule_6_sum = rule_table[2, 6]+ rule_table[2, 5]+rule_table[1, 6]
            # rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out

            self.out = actual_torque + (0.01*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients = [tf.clip_by_value(grad, -0.01, 0.01) for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # print("Gradients for mf_torque:", gradients[self.trainable_variables.index(self.mf_torque)])
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)

    def return_learning_rate(self):
        # print('before:',self.optimizer.learning_rate.numpy())
        return self.optimizer.learning_rate

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
    
        rul_error =tf.exp(-tf.square(tf.subtract(torque, self.mf_torque_predict[1])) / tf.square(self.sigma_torque_predict[1]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [5]), self.mf_delta_torque_predict[1:6])) / tf.square(self.sigma_delta_torque_predict[1:6]))
        

        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("angle:",torque)
        # print("angle mf:",degMF_torque)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[2, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 1]+rule_table[0, 2] + rule_table[1, 2]+rule_table[2, 2]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[1, 5]+rule_table[2, 5]+rule_table[2, 6]+rule_table[0, 6]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[2, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(rul, self.mf_y_predict),self.y_sigma_predict), axis=1)
        den =tf.reduce_sum(tf.multiply(rul,self.y_sigma_predict), axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")

class Torque_ANFIS_multi_1sec:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.25]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y,self.y_sigma]

        self.mf_torque_predict = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.25]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)

        initial_learning_rate = 0.1
        # initial_learning_rate = 0.2
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=False,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )

        self.total_loss = 0
        self.loss_count = 0

    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[2, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 1]+rule_table[0, 2] + rule_table[1, 2]+rule_table[2, 2]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[1, 5]+rule_table[2, 5]+rule_table[2, 6]+rule_table[0, 6]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[2, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out

            self.out = actual_torque + (0.001*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
        
            self.total_loss += loss
            self.loss_count += 1

            if self.loss_count == 3:
                avg_loss = self.total_loss/3
                gradients = tape.gradient(avg_loss, self.trainable_variables)
                # gradients = [tf.clip_by_value(grad, -0.005, 0.005) for grad in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                self.total_loss = 0
                self.loss_count = 0
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)



    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
    
        rul_error =tf.exp(-tf.square(tf.subtract(torque, self.mf_torque_predict[1])) / tf.square(self.sigma_torque_predict[1]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [5]), self.mf_delta_torque_predict[1:6])) / tf.square(self.sigma_delta_torque_predict[1:6]))


        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[2, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 1]+rule_table[0, 2] + rule_table[1, 2]+rule_table[2, 2]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[1, 5]+rule_table[2, 5]+rule_table[2, 6]+rule_table[0, 6]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[2, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(rul, self.mf_y_predict),self.y_sigma_predict), axis=1)
        den =tf.reduce_sum(tf.multiply(rul,self.y_sigma_predict), axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")

class Torque_ANFIS_multi_5fuzzy:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-0.25, -0.15, 0, 0.15, 0.25]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([130, 0.07, 0.07, 0.07, 130]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y,self.y_sigma]

        self.mf_torque_predict = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-0.25, -0.15, 0, 0.15, 0.25]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([130, 0.07, 0.07, 0.07, 130]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -2, 0, 2, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2, 2, 2, 2, 2, 2, 2]), dtype=tf.float64)

        # initial_learning_rate = 0.4
        initial_learning_rate = 0.4
        # initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=True,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(self.torque, self.mf_torque[1])) / tf.square(self.sigma_torque[1]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [3]), self.mf_delta_torque[1:4])) / tf.square(self.sigma_delta_torque[1:4]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[2, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 1]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 2]+rule_table[1, 2]+rule_table[2, 2]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[0, 3]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[1, 3]+rule_table[2, 3]+rule_table[2, 4]+rule_table[0, 4]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[2, 4]
            rule_6 = tf.minimum(rule_6_sum, 1)

            # rule_0_sum = rule_table[2, 0]+rule_table[2, 1]+rule_table[1, 0]
            # rule_0 = tf.minimum(rule_0_sum, 1)
            # rule_1_sum =  rule_table[0, 0] + rule_table[0, 1]+rule_table[1, 1]+rule_table[2, 2]
            # rule_1 = tf.minimum(rule_1_sum, 1)
            # rule_2_sum = rule_table[0, 2]+rule_table[1, 2]
            # rule_2 = tf.minimum(rule_2_sum, 1)
            # rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            # rule_3 = tf.minimum(rule_3_sum, 1)
            # rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            # rule_4 = tf.minimum(rule_4_sum, 1)
            # rule_5_sum = rule_table[0, 5]+rule_table[0, 6]+rule_table[2, 6]+rule_table[2, 4]
            # rule_5 = tf.minimum(rule_5_sum, 1)            
            # rule_6_sum = rule_table[2, 6]+ rule_table[2, 5]+rule_table[1, 6]
            # rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),self.y_sigma), axis=1)
            den =tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out

            self.out = actual_torque + (0.01*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients = [tf.clip_by_value(grad, -0.01, 0.01) for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # print("Gradients for mf_torque:", gradients[self.trainable_variables.index(self.mf_torque)])
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)

    def return_learning_rate(self):
        # print('before:',self.optimizer.learning_rate.numpy())
        return self.optimizer.learning_rate

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
    
        rul_error =tf.exp(-tf.square(tf.subtract(torque, self.mf_torque_predict[1])) / tf.square(self.sigma_torque_predict[1]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [3]), self.mf_delta_torque_predict[1:4])) / tf.square(self.sigma_delta_torque_predict[1:4]))
        

        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("angle:",torque)
        # print("angle mf:",degMF_torque)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[2, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum =  rule_table[0, 0] + rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 1]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 2]+rule_table[1, 2]+rule_table[2, 2]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[0, 3]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[1, 3]+rule_table[2, 3]+rule_table[2, 4]+rule_table[0, 4]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[2, 4]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(tf.multiply(rul, self.mf_y_predict),self.y_sigma_predict), axis=1)
        den =tf.reduce_sum(tf.multiply(rul,self.y_sigma_predict), axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")

class Torque_ANFIS_multi_pos:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y]

        self.mf_torque_predict = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2]), dtype=tf.float64)

        # initial_learning_rate = 0.4
        initial_learning_rate = 0.8
        # initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=True,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.torque, [5]), self.mf_torque[1:6])) / tf.square(self.sigma_torque[1:6]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:6])) / tf.square(self.sigma_delta_torque[1:6]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)

            # rule_0_sum = rule_table[2, 0]+rule_table[2, 1]+rule_table[1, 0]
            # rule_0 = tf.minimum(rule_0_sum, 1)
            # rule_1_sum =  rule_table[0, 0] + rule_table[0, 1]+rule_table[1, 1]+rule_table[2, 2]
            # rule_1 = tf.minimum(rule_1_sum, 1)
            # rule_2_sum = rule_table[0, 2]+rule_table[1, 2]
            # rule_2 = tf.minimum(rule_2_sum, 1)
            # rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            # rule_3 = tf.minimum(rule_3_sum, 1)
            # rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            # rule_4 = tf.minimum(rule_4_sum, 1)
            # rule_5_sum = rule_table[0, 5]+rule_table[0, 6]+rule_table[2, 6]+rule_table[2, 4]
            # rule_5 = tf.minimum(rule_5_sum, 1)            
            # rule_6_sum = rule_table[2, 6]+ rule_table[2, 5]+rule_table[1, 6]
            # rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            # num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),tf.tile(self.y_sigma, [5])), axis=1)
            # den =tf.reduce_sum(tf.multiply(self.rul,tf.tile(self.y_sigma, [5])), axis=1)
            num = tf.reduce_sum(tf.multiply(self.rul, self.mf_y), axis=1)
            den =tf.reduce_sum(self.rul, axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out
            
            self.out = actual_torque + (0.01*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_value(grad, -0.01, 0.01) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # print("Gradients for mf_torque:", gradients[self.trainable_variables.index(self.mf_torque)])
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)

    def return_learning_rate(self):
        # print('before:',self.optimizer.learning_rate.numpy())
        return self.optimizer.learning_rate

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
 
        rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(torque, [5]), self.mf_torque_predict[1:6])) / tf.square(self.sigma_torque_predict[1:6]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [5]), self.mf_delta_torque_predict[1:6])) / tf.square(self.sigma_delta_torque_predict[1:6]))
        

        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("angle:",torque)
        # print("angle mf:",degMF_torque)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[0, 0] + rule_table[0, 1] + rule_table[1, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 2]+rule_table[0, 3]+rule_table[0, 4]+rule_table[1, 1]+rule_table[1, 2]+rule_table[1, 3]+rule_table[2, 0]+rule_table[2, 1]+rule_table[2, 2]+rule_table[3, 0]+rule_table[3, 1]+rule_table[4, 0]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 5]+rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]+rule_table[5, 0]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 6]+rule_table[1, 5]+rule_table[2, 4]+rule_table[3, 3]+rule_table[4, 2]+rule_table[5, 1]+rule_table[6, 0]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[1, 6]+rule_table[2, 5]+rule_table[3, 4]+rule_table[4, 3]+rule_table[5, 2]+rule_table[6, 1]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[2, 6]+rule_table[3, 6]+rule_table[3, 5]+rule_table[4, 6]+rule_table[4, 5]+rule_table[4, 4]+rule_table[5, 5]+rule_table[5, 4]+rule_table[5, 3]+rule_table[6, 4]+rule_table[6, 3]+rule_table[6, 2]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[6, 5]+rule_table[5, 6]+rule_table[6, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(rul, self.mf_y_predict), axis=1)
        den =tf.reduce_sum(rul, axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")

class Torque_ANFIS_multi_pos_5Rule:

    def __init__(self):

        self.mf_torque = tf.Variable(np.asarray([-8.5, -5, 0, 5, 8.5]), dtype=tf.float64)
        self.sigma_torque = tf.Variable(np.asarray([2, 2.5, 2.5, 2.5, 2]), dtype=tf.float64)
        self.mf_delta_torque = tf.Variable(np.asarray([-8.5, -5, 0, 5, 8.5]), dtype=tf.float64)
        self.sigma_delta_torque = tf.Variable(np.asarray([2, 2.5, 2.5, 2.5, 2]), dtype=tf.float64)
        self.mf_y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma = tf.Variable(np.asarray([2]), dtype=tf.float64)
        self.trainable_variables = [self.mf_torque,self.sigma_torque,self.mf_delta_torque,self.sigma_delta_torque,self.mf_y,self.y_sigma]

        self.mf_torque_predict = tf.Variable(np.asarray([-8.5, -5, 0, 5, 8.5]), dtype=tf.float64)
        self.sigma_torque_predict = tf.Variable(np.asarray([2, 2.5, 2.5, 2.5, 2]), dtype=tf.float64)
        self.mf_delta_torque_predict = tf.Variable(np.asarray([-8.5, -5, 0, 5, 8.5]), dtype=tf.float64)
        self.sigma_delta_torque_predict = tf.Variable(np.asarray([2, 2.5, 2.5, 2.5, 2]), dtype=tf.float64)
        self.mf_y_predict = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float64)
        self.y_sigma_predict = tf.Variable(np.asarray([2]), dtype=tf.float64)

        # initial_learning_rate = 0.4
        initial_learning_rate = 0.4
        # initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=125,  # 每 10000 步减少学习率
            decay_rate=0.9,    # 每次减少 4%
            staircase=True      # True 表示学习率阶梯式降低，False 表示平滑降低
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001,   # 默认初始学习率
        #     beta_1=0.9,            # 一阶矩估计的衰减率
        #     beta_2=0.999,          # 二阶矩估计的衰减率
        #     epsilon=1e-07,         # 防止除以0的一个小常数
        #     amsgrad=True,         # 是否使用AMSGrad，适用于某些特殊的优化问题
        #     name='Adam'
        # )


    def train(self, torque, delta_torque, targets_torque, actual_torque):
        with tf.GradientTape() as tape:
            self.torque = tf.convert_to_tensor(torque, dtype=tf.float64)
            self.delta_torque = tf.convert_to_tensor( delta_torque, dtype=tf.float64)
            targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
            actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

            # self.error = self.error * 30
            # self.delta = self.delta * 30
            # targets_torque = targets_torque * 30
            # actual_torque = actual_torque * 30

            self.torque_first = 1 / (1 + tf.exp(self.sigma_torque[0]*(self.torque[0] - self.mf_torque[0])))
            self.delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque[0]*(self.delta_torque[0]  - self.mf_delta_torque[0])))
            self.torque_last = 1 / (1 + tf.exp(-self.sigma_torque[-1]*(self.torque[0]  - self.mf_torque[-1])))
            self.delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque[-1]*(self.delta_torque[0]  - self.mf_delta_torque[-1])))

            self.rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(self.torque, [5]), self.mf_torque[1:4])) / tf.square(self.sigma_torque[1:4]))
            self.rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(self.delta_torque, [5]), self.mf_delta_torque[1:4])) / tf.square(self.sigma_delta_torque[1:4]))

            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last = tf.expand_dims(self.torque_last , -1)
            self.torque_first = tf.expand_dims(self.torque_first, -1)  # 在最后增加一个维度
            self.torque_last  = tf.expand_dims(self.torque_last , -1)
            self.rul_error = tf.expand_dims(self.rul_error, axis=-1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.delta_torque_first = tf.expand_dims(self.delta_torque_first, -1)  # 在最后增加一个维度
            self.delta_torque_last = tf.expand_dims(self.delta_torque_last, -1)
            self.rul_delta = tf.expand_dims(self.rul_delta, axis=-1)


            self.degMF_torque = tf.concat([self.torque_first, self.rul_error, self.torque_last], axis=0)
            self.degMF_delta_torque = tf.concat([self.delta_torque_first, self.rul_delta, self.delta_torque_last], axis=0)

            
            # 使用 tf.maximum 进行逐元素比较
            rule_table  = tf.minimum(tf.expand_dims(self.degMF_torque, axis=1), tf.expand_dims(self.degMF_delta_torque, axis=0))

            rule_0_sum = rule_table[0, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[0, 1]+rule_table[0, 2]+rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[0, 3]+rule_table[1, 2]+rule_table[2, 1]+rule_table[3, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[0, 4]+rule_table[1, 3]+rule_table[2, 2]+rule_table[3, 1]+rule_table[4, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[2, 4]+rule_table[3, 3]+rule_table[3, 4]+rule_table[4, 2]+rule_table[4, 3]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[4, 4]
            rule_6 = tf.minimum(rule_6_sum, 1)

            # rule_0_sum = rule_table[2, 0]+rule_table[2, 1]+rule_table[1, 0]
            # rule_0 = tf.minimum(rule_0_sum, 1)
            # rule_1_sum =  rule_table[0, 0] + rule_table[0, 1]+rule_table[1, 1]+rule_table[2, 2]
            # rule_1 = tf.minimum(rule_1_sum, 1)
            # rule_2_sum = rule_table[0, 2]+rule_table[1, 2]
            # rule_2 = tf.minimum(rule_2_sum, 1)
            # rule_3_sum = rule_table[0, 3]+rule_table[1, 3]+rule_table[2, 3]
            # rule_3 = tf.minimum(rule_3_sum, 1)
            # rule_4_sum = rule_table[0, 4]+rule_table[1, 4]+rule_table[2, 4]+rule_table[0, 5]
            # rule_4 = tf.minimum(rule_4_sum, 1)
            # rule_5_sum = rule_table[0, 5]+rule_table[0, 6]+rule_table[2, 6]+rule_table[2, 4]
            # rule_5 = tf.minimum(rule_5_sum, 1)            
            # rule_6_sum = rule_table[2, 6]+ rule_table[2, 5]+rule_table[1, 6]
            # rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # print(self.rul)
            # Fuzzy base expansion function:
            # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
            # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
            self.rul = tf.where(tf.math.is_nan(self.rul), tf.zeros_like(self.rul), self.rul)
            # num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.mf_y),tf.tile(self.y_sigma, [5])), axis=1)
            # den =tf.reduce_sum(tf.multiply(self.rul,tf.tile(self.y_sigma, [5])), axis=1)
            num = tf.reduce_sum(tf.multiply(self.rul, self.mf_y), axis=1)
            den =tf.reduce_sum(self.rul, axis=1)
            self.out = tf.divide(num, den)
            self.u = self.out
            
            self.out = actual_torque + (0.01*self.out)
 
            # print('rul:',self.rul)
            # print('y:',self.y)
            # print('y_s:',self.y_sigma)
            # print('u',self.u)
            mse_loss = tf.keras.losses.MeanSquaredError()
            # loss = mse_loss(targets_torque, self.out)
            loss = tf.sqrt(mse_loss(targets_torque, self.out))

            if tf.math.is_nan(self.u):
                self.u = 0
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients = [tf.clip_by_value(grad, -0.01, 0.01) for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # print("Gradients for mf_torque:", gradients[self.trainable_variables.index(self.mf_torque)])
        # print("torque",self.degMF_torque)
        # print("delta torque",self.degMF_delta_torque)
        # print("rule",self.rul)

    def return_learning_rate(self):
        # print('before:',self.optimizer.learning_rate.numpy())
        return self.optimizer.learning_rate

    def change_learning_rate(self,value):
        # print('before:',self.optimizer.learning_rate.numpy())
        self.optimizer.learning_rate = value
        # print('after:',self.optimizer.learning_rate.numpy())

    def return_model(self):
        # 获取所有数组的最大长度
        max_length = max(
            self.mf_torque.shape[0], 
            self.sigma_torque.shape[0],
            self.mf_delta_torque.shape[0], 
            self.sigma_delta_torque.shape[0],
            self.mf_y.shape[0], 
            self.y_sigma.shape[0]
        )

        # 使用 np.pad 来填充数组到最大长度
        mf_torque_padded = np.pad(self.mf_torque.numpy(), (0, max_length - self.mf_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_torque_padded = np.pad(self.sigma_torque.numpy(), (0, max_length - self.sigma_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_delta_torque_padded = np.pad(self.mf_delta_torque.numpy(), (0, max_length - self.mf_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        sigma_delta_torque_padded = np.pad(self.sigma_delta_torque.numpy(), (0, max_length - self.sigma_delta_torque.shape[0]), 'constant', constant_values=np.nan)
        mf_y_padded = np.pad(self.mf_y.numpy(), (0, max_length - self.mf_y.shape[0]), 'constant', constant_values=np.nan)
        y_sigma_padded = np.pad(self.y_sigma.numpy(), (0, max_length - self.y_sigma.shape[0]), 'constant', constant_values=np.nan)

        # 将所有数据组合成一个数组
        all_data = np.column_stack((
            mf_torque_padded,
            sigma_torque_padded,
            mf_delta_torque_padded,
            sigma_delta_torque_padded,
            mf_y_padded,
            y_sigma_padded
        ))

        return all_data
    
    def predict(self, torque, delta_torque):
        torque = tf.convert_to_tensor(torque, dtype=tf.float64)
        delta_torque = tf.convert_to_tensor(delta_torque, dtype=tf.float64)
        # targets_torque = tf.convert_to_tensor(targets_torque, dtype=tf.float64)
        # actual_torque = tf.convert_to_tensor(actual_torque, dtype=tf.float64)

        # self.error = self.error * 30
        # self.delta = self.delta * 30
        # targets_torque = targets_torque * 30
        # actual_torque = actual_torque * 30

        torque_first = 1 / (1 + tf.exp(self.sigma_torque_predict[0]*(torque[0] - self.mf_torque_predict[0])))
        delta_torque_first = 1 / (1 + tf.exp(self.sigma_delta_torque_predict[0]*(delta_torque[0]  - self.mf_delta_torque_predict[0])))
        torque_last = 1 / (1 + tf.exp(-self.sigma_torque_predict[-1]*(torque[0]  - self.mf_torque_predict[-1])))
        delta_torque_last = 1 / (1 + tf.exp(-self.sigma_delta_torque_predict[-1]*(delta_torque[0]  - self.mf_delta_torque_predict[-1])))
 
        rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(torque, [5]), self.mf_torque_predict[1:4])) / tf.square(self.sigma_torque_predict[1:4]))
        rul_delta =tf.exp(-tf.square(tf.subtract(tf.tile(delta_torque, [5]), self.mf_delta_torque_predict[1:4])) / tf.square(self.sigma_delta_torque_predict[1:4]))
        

        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last = tf.expand_dims(torque_last , -1)
        torque_first = tf.expand_dims(torque_first, -1)  # 在最后增加一个维度
        torque_last  = tf.expand_dims(torque_last , -1)
        rul_error = tf.expand_dims(rul_error, axis=-1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        delta_torque_first = tf.expand_dims(delta_torque_first, -1)  # 在最后增加一个维度
        delta_torque_last = tf.expand_dims(delta_torque_last, -1)
        rul_delta = tf.expand_dims(rul_delta, axis=-1)

        degMF_torque = tf.concat([torque_first, rul_error, torque_last], axis=0)
        degMF_delta_torque = tf.concat([delta_torque_first, rul_delta, delta_torque_last], axis=0)
        # print("angle:",torque)
        # print("angle mf:",degMF_torque)
        # print("self.degMF_torque pre:",tf.shape(degMF_torque))
        # print("self.degMF_delta_torque pre",tf.shape(degMF_delta_torque))
        # 使用 tf.maximum 进行逐元素比较
        rule_table  = tf.minimum(tf.expand_dims(degMF_torque, axis=1), tf.expand_dims(degMF_delta_torque, axis=0))
        # print("rule_table pre :",tf.shape(rule_table))
        rule_0_sum = rule_table[0, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[0, 1]+rule_table[0, 2]+rule_table[1, 0]+rule_table[1, 1]+rule_table[2, 1]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[0, 3]+rule_table[1, 2]+rule_table[2, 1]+rule_table[3, 0]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[0, 4]+rule_table[1, 3]+rule_table[2, 2]+rule_table[3, 1]+rule_table[4, 0]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[1, 4]+rule_table[2, 3]+rule_table[3, 2]+rule_table[4, 1]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[2, 4]+rule_table[3, 3]+rule_table[3, 4]+rule_table[4, 2]+rule_table[4, 3]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[4, 4]
        rule_6 = tf.minimum(rule_6_sum, 1)
        rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        rul = tf.reshape(rul, (1, 7))
        # print(self.rul)
        # Fuzzy base expansion function:
        # num = tf.clip_by_value(tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1), 1e-6, 1e6)
        # den = tf.clip_by_value(tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1), 1e-6, 1e6)
        rul = tf.where(tf.math.is_nan(rul), tf.zeros_like(rul), rul)
        num = tf.reduce_sum(tf.multiply(rul, self.mf_y_predict), axis=1)
        den =tf.reduce_sum(rul, axis=1)
        out = tf.divide(num, den)
        u = out

        return tf.squeeze(u)

    def update_parameter(self):
        self.mf_torque_predict.assign(self.mf_torque)
        self.sigma_torque_predict.assign(self.sigma_torque)
        self.mf_delta_torque_predict.assign(self.mf_delta_torque)
        self.sigma_delta_torque_predict.assign(self.sigma_delta_torque)
        self.mf_y_predict.assign(self.mf_y)
        self.y_sigma_predict.assign(self.y_sigma)


    def load_model(self, model_data_flat, max_length=7):
        """
        加载保存的模型数据。
        model_data_flat: 通过 np.load() 读取的历史模型数据，是一个一维数组。
        max_length: 模型数据的最大长度，用于 reshape。
        """
        # 如果 model_data_flat 是 1 维数组，将其转换为 2D
        if model_data_flat.ndim == 1:
            # 将扁平化的数据重塑为二维数组，假设每行有 6 个参数
            model_data = model_data_flat.reshape(-1, 6)
        else:
            model_data = model_data_flat

        # Remove NaN values from each column before assigning to the model attributes
        self.mf_torque_predict.assign(model_data[~np.isnan(model_data[:, 0]), 0])  # Remove NaN from mf_torque
        self.sigma_torque_predict.assign(model_data[~np.isnan(model_data[:, 1]), 1])  # Remove NaN from sigma_torque
        self.mf_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 2]), 2])  # Remove NaN from mf_delta_torque
        self.sigma_delta_torque_predict.assign(model_data[~np.isnan(model_data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
        self.mf_y_predict.assign(model_data[~np.isnan(model_data[:, 4]), 4])  # Remove NaN from mf_y
        self.y_sigma_predict.assign(model_data[~np.isnan(model_data[:, 5]), 5])  # Remove NaN from y_sigma

        print("Model loaded successfully without NaN values!")