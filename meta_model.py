
from __future__ import print_function


import tensorflow as tf
import pickle

# Parameters
learning_rate = 0.05
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 3 # 2nd layer number of neurons
n_input = 2 # data input (shape: 1*2)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [7, n_input])
Y = tf.placeholder("float", [1, n_classes])
p_items = tf.placeholder("float",[n_hidden_2,1])
n_items = tf.placeholder("float",[n_hidden_2,1])
test_items = tf.placeholder("float",[1,n_hidden_2])

# Store weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'w0': tf.Variable(tf.random_normal([1, n_classes])),
    'w1': tf.Variable(tf.random_normal([1, n_classes]))

}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_pre': tf.Variable(tf.random_normal([1,n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 10 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 3 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class

    return layer_2

def aggregating_function(f_items,num):
    p_items = tf.div(tf.add_n([f_items[0],f_items[1],f_items[2]]),3)
    n_items = tf.div(tf.add_n([f_items[3],f_items[4],f_items[5]]),3)
    test_items = f_items[6]

    return tf.transpose(p_items),tf.transpose(n_items),test_items
def lwa_model(p_items,n_items,test_items):
    return tf.matmul(test_items,(tf.add(tf.matmul(n_items,weight['w0']),tf.matmul(p_items,weight['w1'])))) + weight['b_pre']


# Construct model
f_items = multilayer_perceptron(X)
p_items,n_items,test_items = aggregating_function(f_items)

logits = lwa_model(p_items,n_items,test_items)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(file_name))
        # Loop over all batches
        for i in file_name:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            x,y = i['x'],i['y']
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer,cost], feed_dict={X: x,
                                                            Y: y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x, Y: y}))