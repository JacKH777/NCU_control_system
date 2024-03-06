import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ANFIS:

    def __init__(self, n_inputs = 1, n_rules = 7, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules

        self.mu_error = tf.Variable(np.tile(np.asarray([-9, -6, -3, 0, 3, 6, 9]), (1)), dtype=tf.float32)
        self.sigma_error = tf.Variable(np.tile(np.asarray([1,1,1,1,1,1,1]), (1)), dtype=tf.float32)
        self.mu_delta = tf.Variable(np.tile(np.asarray([-9, -6, -3, 0, 3, 6, 9]), (1)), dtype=tf.float32)
        self.sigma_delta = tf.Variable(np.tile(np.asarray([1,1,1,1,1,1,1]), (1)), dtype=tf.float32)
        self.y = tf.Variable(np.asarray([-9, -6, -3, 0, 3, 6, 9], dtype=np.float32))
        # self.y_sigma = tf.Variable(np.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5], dtype=np.float32))
        self.trainable_variables = [self.mu_error, self.sigma_error,self.mu_delta, self.sigma_delta, self.y]

        # self.optimizer = tf.optimizers.Adam(0.05) # Optimization step
        self.optimizer = tf.optimizers.RMSprop(learning_rate=0.1, rho=0.8)
        # self.optimizer = tf.optimizers.Adadelta(learning_rate=1, rho=0.9)
    
    def train(self, error,delta, targets, actual_angle):
        with tf.GradientTape() as tape:
            self.error = tf.convert_to_tensor( error, dtype=tf.float32)
            self.delta = tf.convert_to_tensor( delta, dtype=tf.float32)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            actual_angle = tf.convert_to_tensor(actual_angle, dtype=tf.float32)
            self.rul_error = tf.reduce_prod(
            tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [7]), self.mu_error)) / tf.square(self.sigma_error)),
                       (-1,7, 1)), axis=2)  # Rule activations
            self.rul_delta = tf.reduce_prod(
            tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [7]), self.mu_delta)) / tf.square(self.sigma_delta)),
                       (-1,7, 1)), axis=2)  # Rule activations
            
            rul_error_expanded = tf.expand_dims(self.rul_error, 2)  # 扩展为 [-1, 7, 1] 然后复制
            rul_delta_expanded = tf.expand_dims(self.rul_delta, 1)  # 扩展为 [-1, 1, 7] 然后复制

            # 使用 tf.maximum 进行逐元素比较
            rule_table = tf.minimum(tf.tile(rul_error_expanded, [1, 1, 7]), tf.tile(rul_delta_expanded, [1, 7, 1]))

            rule_0_sum = rule_table[:,0, 0] + rule_table[:,0, 1] + rule_table[:,1, 0]
            rule_0 = tf.minimum(rule_0_sum, 1)
            rule_1_sum = rule_table[:,0, 2]+rule_table[:,0, 3]+rule_table[:,0, 4]+rule_table[:,1, 1]+rule_table[:,1, 2]+rule_table[:,1, 3]+rule_table[:,2, 0]+rule_table[:,2, 1]+rule_table[:,2, 2]+rule_table[:,3, 0]+rule_table[:,3, 1]+rule_table[:,4, 0]
            rule_1 = tf.minimum(rule_1_sum, 1)
            rule_2_sum = rule_table[:,0, 5]+rule_table[:,1, 4]+rule_table[:,2, 3]+rule_table[:,3, 2]+rule_table[:,4, 1]+rule_table[:,5, 0]
            rule_2 = tf.minimum(rule_2_sum, 1)
            rule_3_sum = rule_table[:,0, 6]+rule_table[:,1, 5]+rule_table[:,2, 4]+rule_table[:,3, 3]+rule_table[:,4, 2]+rule_table[:,5, 1]+rule_table[:,6, 0]
            rule_3 = tf.minimum(rule_3_sum, 1)
            rule_4_sum = rule_table[:,1, 6]+rule_table[:,2, 5]+rule_table[:,3, 4]+rule_table[:,4, 3]+rule_table[:,5, 2]+rule_table[:,6, 1]
            rule_4 = tf.minimum(rule_4_sum, 1)
            rule_5_sum = rule_table[:,2, 6]+rule_table[:,3, 6]+rule_table[:,3, 5]+rule_table[:,4, 6]+rule_table[:,4, 5]+rule_table[:,4, 4]+rule_table[:,5, 5]+rule_table[:,5, 4]+rule_table[:,5, 3]+rule_table[:,6, 4]+rule_table[:,6, 3]+rule_table[:,6, 2]
            rule_5 = tf.minimum(rule_5_sum, 1)            
            rule_6_sum = rule_table[:,6, 5]+rule_table[:,5, 6]+rule_table[:,6, 6]
            rule_6 = tf.minimum(rule_6_sum, 1)
            self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
            self.rul = tf.reshape(self.rul, (1, 7))
            # Fuzzy base expansion function:
            num = tf.reduce_sum(tf.multiply(self.rul, self.y), axis=1)
            den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
            self.out = tf.divide(num, den)
            self.u = self.out
            self.out = actual_angle + (0.01 * self.out)
            mae_loss = tf.keras.losses.MeanAbsoluteError()
            loss = mae_loss(targets, self.out)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.squeeze(self.u), self.mu_error, self.sigma_error, self.mu_delta, self.sigma_delta, self.y

    def predict(self, error, delta, actual_angle,mu_error, sigma_error,mu_delta, sigma_delta, y):
        self.error = tf.convert_to_tensor(error, dtype=tf.float32)
        self.delta = tf.convert_to_tensor(delta, dtype=tf.float32)
        actual_angle = tf.convert_to_tensor(actual_angle, dtype=tf.float32)
        self.mu_error = mu_error
        self.sigma_error = sigma_error
        self.mu_delta = mu_delta
        self.sigma_delta = sigma_delta
        self.y = y
        self.rul_error = tf.reduce_prod(
            tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.error, [7]), self.mu_error)) / tf.square(self.sigma_error)),
                    (-1, 7, 1)), axis=2)  # Rule activations
        self.rul_delta = tf.reduce_prod(
            tf.reshape(tf.exp(-tf.square(tf.subtract(tf.tile(self.delta, [7]), self.mu_delta)) / tf.square(self.sigma_delta)),
                    (-1, 7, 1)), axis=2)  # Rule activations

        rul_error_expanded = tf.expand_dims(self.rul_error, 2)  # 扩展为 [-1, 7, 1] 然后复制
        rul_delta_expanded = tf.expand_dims(self.rul_delta, 1)  # 扩展为 [-1, 1, 7] 然后复制

        # 使用 tf.maximum 进行逐元素比较
        rule_table = tf.minimum(tf.tile(rul_error_expanded, [1, 1, 7]), tf.tile(rul_delta_expanded, [1, 7, 1]))

        rule_0_sum = rule_table[:,0, 0] + rule_table[:,0, 1] + rule_table[:,1, 0]
        rule_0 = tf.minimum(rule_0_sum, 1)
        rule_1_sum = rule_table[:,0, 2]+rule_table[:,0, 3]+rule_table[:,0, 4]+rule_table[:,1, 1]+rule_table[:,1, 2]+rule_table[:,1, 3]+rule_table[:,2, 0]+rule_table[:,2, 1]+rule_table[:,2, 2]+rule_table[:,3, 0]+rule_table[:,3, 1]+rule_table[:,4, 0]
        rule_1 = tf.minimum(rule_1_sum, 1)
        rule_2_sum = rule_table[:,0, 5]+rule_table[:,1, 4]+rule_table[:,2, 3]+rule_table[:,3, 2]+rule_table[:,4, 1]+rule_table[:,5, 0]
        rule_2 = tf.minimum(rule_2_sum, 1)
        rule_3_sum = rule_table[:,0, 6]+rule_table[:,1, 5]+rule_table[:,2, 4]+rule_table[:,3, 3]+rule_table[:,4, 2]+rule_table[:,5, 1]+rule_table[:,6, 0]
        rule_3 = tf.minimum(rule_3_sum, 1)
        rule_4_sum = rule_table[:,1, 6]+rule_table[:,2, 5]+rule_table[:,3, 4]+rule_table[:,4, 3]+rule_table[:,5, 2]+rule_table[:,6, 1]
        rule_4 = tf.minimum(rule_4_sum, 1)
        rule_5_sum = rule_table[:,2, 6]+rule_table[:,3, 6]+rule_table[:,3, 5]+rule_table[:,4, 6]+rule_table[:,4, 5]+rule_table[:,4, 4]+rule_table[:,5, 5]+rule_table[:,5, 4]+rule_table[:,5, 3]+rule_table[:,6, 4]+rule_table[:,6, 3]+rule_table[:,6, 2]
        rule_5 = tf.minimum(rule_5_sum, 1)            
        rule_6_sum = rule_table[:,6, 5]+rule_table[:,5, 6]+rule_table[:,6, 6]
        rule_6 = tf.minimum(rule_6_sum, 1)
        self.rul = tf.stack([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6])
        self.rul = tf.reshape(self.rul, (1, 7))
        # Fuzzy base expansion function:
        num = tf.reduce_sum(tf.multiply(self.rul, self.y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)
        self.u = self.out
        self.out = actual_angle + (0.01 * self.out)
        return tf.squeeze(self.u)
        