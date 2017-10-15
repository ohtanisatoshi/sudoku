import numpy as np
import tensorflow as tf

q = np.array([[0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,2,7,
              4,0,0,6,0,8,0,0,0,
              0,7,1,0,0,0,3,0,0,
              2,3,8,5,0,6,4,1,9,
              9,6,4,1,0,0,7,5,0,
              3,9,5,0,2,7,8,0,0,
              1,8,2,0,6,0,9,7,4,
              0,4,6,8,1,9,2,0,5
              ]]).astype(np.float32)
a = np.array([[6,1,9,7,3,2,5,4,8,
              8,5,3,9,4,1,6,2,7,
              4,2,7,6,5,8,1,9,3,
              5,7,1,2,9,4,3,8,6,
              2,3,8,5,7,6,4,1,9,
              9,6,4,1,8,3,7,5,2,
              3,9,5,4,2,7,8,6,1,
              1,8,2,3,6,5,9,7,4,
              7,4,6,8,1,9,2,3,5
              ]]).astype(np.float32)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 81])
y_ = tf.placeholder(tf.float32, [None, 81])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

w_fc1 = weight_variable([81, 81])
b_fc1 = bias_variable([81])
h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
w_fc2 = weight_variable([81, 81])
b_fc2 = bias_variable([81])
h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2

loss = tf.reduce_mean(tf.square(h_fc2 - y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())
for i in range(1000):
    train_step.run(feed_dict={x: q, y_:a})
    if i % 10 == 0:
        l = sess.run(loss, feed_dict={x: q, y_:a})
        print(l)


result = sess.run(h_fc2, feed_dict={x: q})
for i, r in enumerate(result[0]):
    print('{:.0f},'.format(round(r)), end="")
    if (i+1) % 9 == 0:
        print('\n')


