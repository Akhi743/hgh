"""GANITE Codebase.

ganite.py

Note: GANITE module adapted for LaLonde dataset with proper scaling.
"""

# Necessary packages
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import numpy as np
from utils import xavier_init, batch_generator

def ganite(train_x, train_t, train_y, test_x, parameters):
    """GANITE module.
    
    Args:
        - train_x: features in training data
        - train_t: treatments in training data
        - train_y: observed outcomes in training data
        - test_x: features in testing data
        - parameters: GANITE network parameters
            - h_dim: hidden dimensions
            - batch_size: the number of samples in each batch
            - iterations: the number of iterations for training
            - alpha: hyper-parameter to adjust the loss importance
            
    Returns:
        - test_y_hat: estimated potential outcomes
    """
    # Parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    alpha = parameters['alpha']
    
    # Scale the outcomes to [0,1] range for better training stability
    y_max = np.max(train_y)
    train_y_scaled = train_y / y_max
    
    no, dim = train_x.shape
    
    # Reset graph
    tf.reset_default_graph()
    
    ## Placeholders
    X = tf.placeholder(tf.float32, shape=[None, dim])
    T = tf.placeholder(tf.float32, shape=[None, 1])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    ## Generator Variables
    G_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b31 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W32 = tf.Variable(xavier_init([h_dim, 1]))
    G_b32 = tf.Variable(tf.zeros(shape=[1]))
    G_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b41 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W42 = tf.Variable(xavier_init([h_dim, 1]))
    G_b42 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, 
               G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]
    
    ## Discriminator Variables
    D_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, 1]))
    D_b3 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
    ## Inference Variables
    I_W1 = tf.Variable(xavier_init([dim, h_dim]))
    I_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    I_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    I_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b31 = tf.Variable(tf.zeros(shape=[h_dim]))
    I_W32 = tf.Variable(xavier_init([h_dim, 1]))
    I_b32 = tf.Variable(tf.zeros(shape=[1]))
    I_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
    I_b41 = tf.Variable(tf.zeros(shape=[h_dim]))
    I_W42 = tf.Variable(xavier_init([h_dim, 1]))
    I_b42 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_I = [I_W1, I_W2, I_W31, I_W32, I_W41, I_W42,
               I_b1, I_b2, I_b31, I_b32, I_b41, I_b42]
    
    def generator(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        
        G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
        G_logit1 = tf.nn.sigmoid(tf.matmul(G_h31, G_W32) + G_b32)
        
        G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
        G_logit2 = tf.nn.sigmoid(tf.matmul(G_h41, G_W42) + G_b42)
        
        G_logit = tf.concat(axis=1, values=[G_logit1, G_logit2])
        return G_logit
    
    def discriminator(x, t, y, hat_y):
        input0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1])
        input1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1])
        
        inputs = tf.concat(axis=1, values=[x, input0, input1])
        
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        
        return D_logit
    
    def inference(x):
        I_h1 = tf.nn.relu(tf.matmul(x, I_W1) + I_b1)
        I_h2 = tf.nn.relu(tf.matmul(I_h1, I_W2) + I_b2)
        
        I_h31 = tf.nn.relu(tf.matmul(I_h2, I_W31) + I_b31)
        I_logit1 = tf.nn.sigmoid(tf.matmul(I_h31, I_W32) + I_b32)
        
        I_h41 = tf.nn.relu(tf.matmul(I_h2, I_W41) + I_b41)
        I_logit2 = tf.nn.sigmoid(tf.matmul(I_h41, I_W42) + I_b42)
        
        I_logit = tf.concat(axis=1, values=[I_logit1, I_logit2])
        return I_logit
    
    # Generation
    Y_tilde = generator(X, T, Y)
    D_logit = discriminator(X, T, Y, Y_tilde)
    Y_hat = inference(X)
    
    # Loss
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=T, logits=D_logit))
    
    G_loss_GAN = -D_loss
    
    G_loss_Factual = tf.reduce_mean(
        tf.square(Y - (T * tf.reshape(Y_tilde[:,1], [-1,1]) + 
                      (1. - T) * tf.reshape(Y_tilde[:,0], [-1,1]))))
    
    G_loss = G_loss_Factual + alpha * G_loss_GAN
    
    I_loss = tf.reduce_mean(
        tf.square(Y - (T * tf.reshape(Y_hat[:,1], [-1,1]) + 
                      (1. - T) * tf.reshape(Y_hat[:,0], [-1,1]))))
    
    # Solver
    G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=theta_G)
    D_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss, var_list=theta_D)
    I_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(I_loss, var_list=theta_I)
    
    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training
    print('Start training Generator and Discriminator')
    for it in range(iterations):
        for _ in range(2):
            X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y_scaled, batch_size)
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y_scaled, batch_size)
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) + 
                  ', D loss: ' + str(np.round(D_loss_curr, 4)) + 
                  ', G loss: ' + str(np.round(G_loss_curr, 4)))
    
    print('Start training Inference network')
    for it in range(iterations):
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y_scaled, batch_size)
        _, I_loss_curr = sess.run(
            [I_solver, I_loss],
            feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        if it % 1000 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) + 
                  ', I loss: ' + str(np.round(I_loss_curr, 4)))
    
    # Generate potential outcomes
    test_y_hat = sess.run(Y_hat, feed_dict={X: test_x})
    
    # Scale back the predictions
    test_y_hat = test_y_hat * y_max
    
    return test_y_hat