import tensorflow as tf
from tensorflow.keras import Model, Layer

from cvnn.activations import zrelu, modrelu


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
class ComplexInput(Layer):
    def __init__(self, input_dim):
        super(ComplexInput, self).__init__()
        self.input_dim = input_dim

    def call(self, inputs):
        return tf.complex(tf.math.real(inputs), tf.math.imag(inputs))

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, activation=None):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

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
        x = tf.complex(tf.math.real(x), tf.math.imag(x))

        x = tf.matmul(x, W) + b
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
        self.input_layer = ComplexInput(input_dim)  # Ensure complex input
        self.fc1 = DenseLayer(input_dim, hidden_dim, activation='z_mod_relu')
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, activation='z_mod_relu')
        self.fc3 = DenseLayer(hidden_dim, hidden_dim, activation='z_mod_relu')
        self.fc4 = DenseLayer(hidden_dim, hidden_dim, activation='z_mod_relu')
        self.fc5 = DenseLayer(hidden_dim, output_dim)

    def call(self, inputs):
        x = self.input_layer(inputs)  # Preserve complex values
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    

# class ComplexSGD(tf.keras.optimizers.Optimizer):
#     def __init__(self, learning_rate=0.01, name="ComplexSGD", **kwargs):
#         super().__init__(learning_rate, **kwargs)
#         self.name = name

#     def apply_gradients(self, grads_and_vars):
#         updated_vars = []
#         for grad, var in grads_and_vars:
#             if grad is not None:
#                 real_update = var - self.learning_rate * tf.math.real(grad)
#                 # imag_update = var - self.learning_rate * tf.math.imag(grad) * 1j
#                 imag_update = tf.complex(tf.math.real(var), tf.math.imag(var) - self.learning_rate * tf.math.imag(grad))
#                 # updated_vars.append(var.assign(tf.complex(real_update, imag_update)))
#                 updated_vars.append(var.assign(tf.complex(tf.cast(real_update, tf.float32), tf.cast(imag_update, tf.float32))))
#         return updated_vars
    

# def custom_complex_mse(y_true, y_pred):
#     diff = y_true - y_pred
#     loss = tf.reduce_mean(tf.square(tf.abs(diff)))  # Compute mean squared magnitude
#     return tf.cast(loss, tf.complex64)  # Ensure loss remains complex



