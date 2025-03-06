import tensorflow as tf
from tensorflow.keras import Model, Layer

import cvnn
from cvnn.activations import zrelu, modrelu
from cvnn.losses import ComplexAverageCrossEntropy, ComplexMeanSquareError


#%% Version Pytorch
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, sigma_tilde):
#         # Concatenation de X et Sigma_tilde
#         x = torch.cat((x.view(x.size(0), -1), sigma_tilde.view(sigma_tilde.size(0), -1)), dim=1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x
    

#%% Version Tensorflow
class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, activation=None):
        super(DenseLayer, self).__init__()
        
        # Create separate real and imaginary parts
        real_init = tf.keras.initializers.RandomNormal(stddev=0.1)
        imag_init = tf.keras.initializers.RandomNormal(stddev=0.1)
        
        # Initialize real and imaginary weights separately
        self.W_real = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=real_init,
            dtype=tf.float32,
            trainable=True,
        )
        self.W_imag = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=imag_init,
            dtype=tf.float32,
            trainable=True,
        )
        
        # Combine into a complex weight
        self.W = tf.complex(self.W_real, self.W_imag)

        # Initialize complex biases similarly
        self.b_real = self.add_weight(
            shape=(output_dim,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32,
            trainable=True,
        )
        self.b_imag = self.add_weight(
            shape=(output_dim,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32,
            trainable=True,
        )
        
        self.b = tf.complex(self.b_real, self.b_imag)

        self.activation = activation
    
    def call(self, x):
        W = tf.complex(self.W_real, self.W_imag)  
        b = tf.complex(self.b_real, self.b_imag)  
        
        x = tf.matmul(tf.cast(x, tf.complex64), W) + b
        if self.activation == 'z_mod_relu':
            x = zrelu(x)
            x = modrelu(x)
        elif self.activation == 'zrelu':
            x = zrelu(x)
        elif self.activation == 'modrelu':
            x = modrelu(x)
        return x


    
class NeuralNetwork(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = DenseLayer(input_dim, hidden_dim, activation='z_mod_relu')
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, activation='z_mod_relu')
        self.fc3 = DenseLayer(hidden_dim, output_dim)

    def call(self, inputs):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    

def custom_complex_mse(y_true, y_pred):
    diff = y_true - y_pred
    loss = tf.reduce_mean(tf.square(tf.abs(diff)))  # Compute mean squared magnitude
    return tf.cast(loss, tf.complex64)  # Ensure loss remains complex



