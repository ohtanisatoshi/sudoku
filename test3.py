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
q_mask = q.astype(np.bool).astype(np.float32)
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
x_mask = tf.placeholder(tf.float32, [None, 81])
#y_ = tf.placeholder(tf.float32, [None, 81])
#y_ = tf.constant([36, 36, 36, 36, 36, 36, 36, 36, 36])
y_sum = tf.constant([45., 45., 45., 45., 45., 45., 45., 45., 45.])
y_min = tf.constant([1., 1., 1., 1., 1., 1., 1., 1., 1.])
y_max = tf.constant([9., 9., 9., 9., 9., 9., 9., 9., 9.])
y_prod = tf.constant([362880., 362880., 362880., 362880., 362880., 362880., 362880., 362880., 362880.])
unique_size = tf.constant([9., 9., 9., 9., 9., 9., 9., 9., 9.])
'''
mask = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2,
                    0, 0, 0, 1, 1, 1, 2, 2, 2,
                    0, 0, 0, 1, 1, 1, 2, 2, 2,
                    3, 3, 3, 4, 4, 4, 5, 5, 5,
                    3, 3, 3, 4, 4, 4, 5, 5, 5,
                    3, 3, 3, 4, 4, 4, 5, 5, 5,
                    6, 6, 6, 7, 7, 7, 8, 8, 8,
                    6, 6, 6, 7, 7, 7, 8, 8, 8,
                    6, 6, 6, 7, 7, 7, 8, 8, 8])
'''
mask = tf.constant([0, 1, 2,
                    0, 1, 2,
                    0, 1, 2,
                    3, 4, 5,
                    3, 4, 5,
                    3, 4, 5,
                    6, 7, 8,
                    6, 7, 8,
                    6, 7, 8])
mask1 = tf.constant([1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask2 = tf.constant([0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask3 = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask4 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask5 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask6 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0])
mask7 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 0, 0, 0, 0, 0, 0])
mask8 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 0, 0, 0])
mask9 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

w_fc1 = weight_variable([81, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
w_fc2 = weight_variable([100, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
w_fc3 = weight_variable([100, 100])
b_fc3 = bias_variable([100])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)
w_fc10 = weight_variable([100, 81])
b_fc10 = bias_variable([81])
h_fc10 = tf.matmul(h_fc3, w_fc10) + b_fc10
#h_fc10_work = tf.subtract(h_fc10, tf.reduce_min(h_fc10))
#h_fc10_norm = tf.div(h_fc10_work, tf.reduce_max(h_fc10_work))
#h_fc10_0_to_9 = tf.add(tf.multiply(h_fc10_norm, 8.), 1.)
h_fc10_0_to_9 = tf.add(tf.multiply(tf.sigmoid(h_fc10), 8.), 1.)

h_x_mask = tf.multiply(h_fc10_0_to_9, x_mask)
#h_fc2_9x9x9 = tf.reshape(h_fc2, [9,9,9])
#h_fc2_9x9 = tf.argmax(h_fc2_9x9x9, axis=2, output_type=tf.int32)
h_9x9 = tf.reshape(h_fc10_0_to_9, [9, 9])
sum_0 = tf.reduce_sum(h_9x9, 0)
sum_1 = tf.reduce_sum(h_9x9, 1)
mask_sum = tf.reduce_sum(tf.unsorted_segment_sum(tf.reshape(h_fc10_0_to_9, [27, 3]), mask, 9), axis=1)
prod_0 = tf.log(tf.square(tf.reduce_prod(h_9x9, 0) - y_prod))
prod_1 = tf.log(tf.square(tf.reduce_prod(h_9x9, 1) - y_prod))

loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(tf.square(h_x_mask - x), [9, 9]), axis=1) + \
                      tf.square(sum_0 - y_sum) + \
                      tf.square(sum_1 - y_sum) + \
                      tf.square(mask_sum - y_sum) + \
                      tf.reduce_mean(prod_0) + \
                      tf.reduce_mean(prod_1))
train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

sess.run(tf.global_variables_initializer())
for i in range(100000):
    train_step.run(feed_dict={x: q, x_mask: q_mask})
    if i % 100 == 0:
        l = sess.run(loss, feed_dict={x: q, x_mask:q_mask})
        print(l)


result = sess.run(h_fc10_0_to_9, feed_dict={x: q})
for i, r in enumerate(result[0]):
    print('{:.0f},'.format(round(r)), end="")
    if (i+1) % 9 == 0:
        print('\n')

result = np.reshape(result, [9, 9])
print(np.sum(result, axis=0))
print(np.sum(result, axis=1))

result = sess.run(mask_sum, feed_dict={x: q})
print(result)
result = sess.run(r1, feed_dict={x: q})
print(result)
result = sess.run(r1_u, feed_dict={x: q})
print(result)
result = sess.run(r1_u_size, feed_dict={x: q})
print(result)
