import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ANFIS:

    def __init__(self,learning_rate=0.005):

        self.mu_error = tf.Variable(np.asarray([-5, -3, -1, 0, 1, 3, 5]), dtype=tf.float32)
        self.sigma_error = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
        self.mu_delta = tf.Variable(np.asarray([-5, -3, -1, 0, 1, 3, 5]), dtype=tf.float32)
        self.sigma_delta = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
        self.y = tf.Variable(np.asarray([-10, -5, -3, 0, 3, 5, 10]), dtype=tf.float32)
        self.y_sigma = tf.Variable(np.asarray([1.5, 1, 1, 1, 1, 1, 1.5]), dtype=tf.float32)
        self.trainable_variables = [self.mu_error, self.sigma_error,self.mu_delta, self.sigma_delta, self.y,self.y_sigma]


        initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps= 50,
            decay_rate=0.6,
            staircase=True)
        # self.optimizer = tf.optimizers.Adam(learning_rate = lr_schedule) # Optimization step
        self.optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9)
        # self.optimizer = tf.optimizers.Adadelta(learning_rate=1, rho=0.9)
        # self.optimizer = tf.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    

    def train(self, error,delta, targets, actual_angle):
        with tf.GradientTape() as tape:
            self.error = tf.convert_to_tensor( error, dtype=tf.float32)
            self.delta = tf.convert_to_tensor( delta, dtype=tf.float32)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            actual_angle = tf.convert_to_tensor(actual_angle, dtype=tf.float32)
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
            self.out = actual_angle + (0.001*self.out)
            mae_loss = tf.keras.losses.MeanAbsoluteError()
            loss = mae_loss(targets, self.out)
        if tf.math.is_nan(self.out):
            self.out = 0
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.squeeze(self.u), self.mu_error, self.sigma_error, self.mu_delta, self.sigma_delta, self.y

    def predict(self, error, delta):
        self.error = tf.convert_to_tensor( error, dtype=tf.float32)
        self.delta = tf.convert_to_tensor( delta, dtype=tf.float32)

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
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.rul, self.y),self.y_sigma), axis=1)
        den = tf.reduce_sum(tf.multiply(self.rul,self.y_sigma), axis=1)
        self.out = tf.divide(num, den)
        self.u = self.out

        return tf.squeeze(self.u)
        