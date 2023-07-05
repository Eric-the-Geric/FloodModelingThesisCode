import tensorflow as tf
from tensorflow import keras

class CustomMSE_Count_Zero(keras.losses.Loss):
    def __init__(self, name="custom_mse_count0"):
        super().__init__(name=name)
        

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        error = y_true - y_pred[...,0]
        squared_error = tf.square(error)
        # Create masks for zero and non-zero elements in y_true
        zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        non_zero_mask = 1 - zero_mask

        # number_of_zeros, number_of_non_zeros = self._count_zeros_non_zeros(y_true)
        number_of_zeros = tf.math.count_nonzero(y_true==0,)

        number_of_non_zeros = tf.math.count_nonzero(y_true!=0)
        
        total_number_of_elements = number_of_zeros + number_of_non_zeros
        
        non_zero_weight = number_of_zeros / total_number_of_elements
        
        zero_weight = number_of_non_zeros / total_number_of_elements

        # Apply weights to the squared error based on the masks
        weighted_squared_error = tf.cast(squared_error, dtype=tf.float32) * (tf.cast(non_zero_mask, tf.float32) * tf.cast(non_zero_weight, dtype=tf.float32) + tf.cast(zero_mask, dtype=tf.float32) * tf.cast(zero_weight, dtype=tf.float32))
 
        # Calculate the mean of the weighted squared errors
        loss = tf.reduce_mean(weighted_squared_error)
        return loss*10000

class CustomMAE_Count_Zero(keras.losses.Loss):
    def __init__(self, name="custom_mae_count0"):
        super().__init__(name=name)
        

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        error = y_true - y_pred[...,0]
        squared_error = tf.abs(error)
        # Create masks for zero and non-zero elements in y_true
        zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        non_zero_mask = 1 - zero_mask

        # number_of_zeros, number_of_non_zeros = self._count_zeros_non_zeros(y_true)
        number_of_zeros = tf.math.count_nonzero(y_true==0,)

        number_of_non_zeros = tf.math.count_nonzero(y_true!=0)
        
        total_number_of_elements = number_of_zeros + number_of_non_zeros
        
        non_zero_weight = number_of_zeros / total_number_of_elements
        
        zero_weight = number_of_non_zeros / total_number_of_elements

        # Apply weights to the squared error based on the masks
        weighted_squared_error = tf.cast(squared_error, dtype=tf.float32) * (tf.cast(non_zero_mask, tf.float32) * tf.cast(non_zero_weight, dtype=tf.float32) + tf.cast(zero_mask, dtype=tf.float32) * tf.cast(zero_weight, dtype=tf.float32))
 
        # Calculate the mean of the weighted squared errors
        loss = tf.reduce_mean(weighted_squared_error)
        return loss*1000
    
class CustomMSE_Own_Weight(keras.losses.Loss):
    def __init__(self, name="custom_mse_own", zero_weight=0.9, non_zero_weight=0.1):
        super().__init__(name=name)
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight

 # so it stops printing warnings
        
    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        
        error = y_true - y_pred[...,0]
        squared_error = tf.square(error)
        # Create masks for zero and non-zero elements in y_true
        zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        non_zero_mask = 1 - zero_mask
        # Apply weights to the squared error based on the masks
        weighted_squared_error = squared_error * ((zero_mask * self.zero_weight )+ (non_zero_mask * self.non_zero_weight))
        # Calculate the mean of the weighted squared errors
        loss = tf.reduce_mean(weighted_squared_error, axis=-1)
        return loss*10000

class CustomMAE_Own_Weight(keras.losses.Loss):
    def __init__(self, name="custom_mae_own", zero_weight=0.9, non_zero_weight=0.1):
        super().__init__(name=name)
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight

    @tf.autograph.experimental.do_not_convert# so it stops printing warnings
    def call(self, y_true, y_pred):
        
        error = y_true - y_pred[...,0]
        squared_error = tf.abs(error)
        # Create masks for zero and non-zero elements in y_true
        zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        non_zero_mask = 1 - zero_mask
        # Apply weights to the squared error based on the masks
        weighted_squared_error = squared_error * ((zero_mask *self.zero_weight )+ (non_zero_mask * self.non_zero_weight))
        # Calculate the mean of the weighted squared errors
        loss = tf.reduce_mean(weighted_squared_error, axis=-1)
        return loss*1000
    