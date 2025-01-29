"""GANITE Implementation for LaLonde Dataset.

Core GANITE implementation with modifications for better handling of the LaLonde dataset.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
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
    
    no, dim = train_x.shape
    
    # Reset graph
    tf.reset_default_graph()
    
    ## Placeholders
    X = tf.placeholder(tf.float32, shape=[None, dim])
    T = tf.placeholder(tf.float32, shape=[None, 1])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    ## Generator Variables
    G_W1 = tf.Variable(xavier_init([dim+2, h_dim]))  # dim + treatment + outcome
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, 2]))  # Output both potential outcomes
    G_b3 = tf.Variable(tf.zeros(shape=[2]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    ## Discriminator Variables
    D_W1 = tf.Variable(xavier_init([dim+3, h_dim]))  # dim + factual + counterfactual + treatment
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
    I_W3 = tf.Variable(xavier_init([h_dim, 2]))  # Output both potential outcomes
    I_b3 = tf.Variable(tf.zeros(shape=[2]))
    
    theta_I = [I_W1, I_W2, I_W3, I_b1, I_b2, I_b3]
    
    def generator(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])  # [batch_size, dim+2]
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
        G_out = tf.matmul(G_h2, G_W3) + G_b3  # [batch_size, 2]
        return G_out
    
    def discriminator(x, t, y, g_sample):
        # Select factual outcome from generator output based on treatment
        y_cf = tf.where(tf.equal(t, 1),
                       tf.reshape(g_sample[:, 0], [-1, 1]),  # If t=1, use y(0) as counterfactual
                       tf.reshape(g_sample[:, 1], [-1, 1]))  # If t=0, use y(1) as counterfactual
        
        # Combine all inputs: [x, y, y_cf, t]
        inputs = tf.concat(axis=1, values=[x, y, y_cf, t])  # [batch_size, dim+3]
        
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h1 = tf.nn.dropout(D_h1, keep_prob=0.8)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_h2 = tf.nn.dropout(D_h2, keep_prob=0.8)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit
    
    def inference(x):
        I_h1 = tf.nn.relu(tf.matmul(x, I_W1) + I_b1)
        I_h1 = tf.nn.dropout(I_h1, keep_prob=0.8)
        I_h2 = tf.nn.relu(tf.matmul(I_h1, I_W2) + I_b2)
        I_h2 = tf.nn.dropout(I_h2, keep_prob=0.8)
        I_out = tf.matmul(I_h2, I_W3) + I_b3
        return I_out
    
    # Generator
    G_sample = generator(X, T, Y)
    # Discriminator
    D_prob, D_logit = discriminator(X, T, Y, G_sample)
    # Inference
    I_sample = inference(X)
    
    # Loss
    # 1. Discriminator loss
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit, labels=T))
    D_loss = D_loss_real
    
    # 2. Generator loss
    G_loss_fake = -D_loss_real
    # Factual loss: compare generated outcome with observed outcome for actual treatment
    G_loss_factual = tf.reduce_mean(tf.square(Y - tf.reduce_sum(G_sample * tf.concat(
        [1-T, T], axis=1), axis=1, keepdims=True)))
    G_loss = G_loss_factual + alpha * G_loss_fake
    
    # 3. Inference loss
    I_loss_factual = tf.reduce_mean(tf.square(Y - tf.reduce_sum(I_sample * tf.concat(
        [1-T, T], axis=1), axis=1, keepdims=True)))
    I_loss = I_loss_factual
    
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
            X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        if it % 1000 == 0:
            print(f'Iteration: {it}/{iterations}, ' + 
                  f'D loss: {D_loss_curr:.4f}, ' + 
                  f'G loss: {G_loss_curr:.4f}')
    
    print('Start training Inference network')
    for it in range(iterations):
        X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)
        _, I_loss_curr = sess.run(
            [I_solver, I_loss],
            feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
        
        if it % 1000 == 0:
            print(f'Iteration: {it}/{iterations}, ' + 
                  f'I loss: {I_loss_curr:.4f}')
    
    # Generate potential outcomes
    test_y_hat = sess.run(I_sample, feed_dict={X: test_x})
    
    sess.close()
    
    return test_y_hat