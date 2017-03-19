#!/disk/scratch/mlp/miniconda2/bin/python
import os
import sys
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider, MNISTDataProvider
from mlp.local_foo import get_err_and_acc, fully_connected_layer

# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

nonlinear_arr = 'tf.nn.relu'
learning_rate = 0.001
num_epoch = 100
dropout = 0.75 # Dropout, probability to keep units
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
commands = {
    'tf.nn.relu' : tf.nn.relu,
    'tf.tanh' : tf.tanh,
    'tf.nn.crelu' : tf.nn.crelu,
    'tf.nn.relu6' : tf.nn.relu6,
    'tf.sigmoid' : tf.sigmoid,
    'tf.nn.softplus' : tf.nn.softplus,
    'tf.nn.softsign' : tf.nn.softsign
}

graph = tf.Graph()
if (str(sys.argv[1])=='10'):
    train_data = CIFAR10DataProvider('train', batch_size=50)
    valid_data = CIFAR10DataProvider('valid', batch_size=50)
    dataset ='C10'
if (str(sys.argv[1])=='100'):
    train_data = CIFAR100DataProvider('train', batch_size=50)
    valid_data = CIFAR100DataProvider('valid', batch_size=50)
    dataset ='C100'

train_data.inputs = train_data.inputs.reshape((-1, 1024, 3), order='F')
train_data.inputs = train_data.inputs.reshape((-1,32,32,3))
valid_data.inputs = valid_data.inputs.reshape((-1, 1024, 3), order='F')
valid_inputs = valid_data.inputs.reshape((-1,32,32,3))
valid_targets = valid_data.to_one_of_k(valid_data.targets)

print(train_data.inputs.shape[1])
print(train_data.inputs.shape[2])
print(train_data.inputs.shape[3])

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=3, s=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding='SAME')

def norm(layer, lsize) :
    return tf.nn.lrn(layer, lsize, bias=2.0, alpha=0.001 / 9.0, beta=0.75)



# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc0': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=.1)),
    'wc1b': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 128, 192], stddev=.1)),
    'wc2b': tf.Variable(tf.truncated_normal([3, 3, 192, 240], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 240, 260], stddev=.1)),
    'wc3b': tf.Variable(tf.truncated_normal([3, 3, 260, 280], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 280, 280], stddev=.1)),
    'wc4b': tf.Variable(tf.truncated_normal([3, 3, 280, 300], stddev=.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    #'wc5': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=.1)),
    #'wc5a': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=.1)),
    #'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384]))
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([4*4*64, 4096])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([1200, 300], stddev=2. / (1200 + 300)**0.5)),
    'wd2': tf.Variable(tf.truncated_normal([300, 100], stddev=2. / (300 + 1024)**0.5)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([100, train_data.num_classes]))
}

biases = {
    'bc0': tf.Variable(tf.random_normal([32])),
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([192])),
    'bc4': tf.Variable(tf.random_normal([240])),
    'bd1': tf.Variable(tf.random_normal([260])),
    'bd2': tf.Variable(tf.random_normal([280])),
    'bd3': tf.Variable(tf.random_normal([300])),
    'bd4': tf.Variable(tf.random_normal([100])),
    'out': tf.Variable(tf.random_normal([train_data.num_classes]))
}


with tf.name_scope('data'):
    inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1], train_data.inputs.shape[2], train_data.inputs.shape[3]], 'inputs')

    #x = tf.reshape(inputs, shape=[-1, 32, 32, 3])
    targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
    
    #re = train_data.inputs.reshape((40000,32,32,3))
    #re = re.reshape((40000,32,32,3))

    #print re.shape
    #plt.clf()
    #fig = plt.gcf()
    #for e in range(1):
    #    plt.clf
    #    plt.imshow(re[e])
    #    plt.show
    #    fig.savefig('image.jpg')

########################
#Layer 1 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-1'):
    conv1 = conv2d(inputs, weights['wc0'], biases['bc0'])
    print "conv1.shape:", conv1.get_shape()
    h_pool_conv1 = maxpool2d(conv1, k=2)
    print "h_pool_conv1.shape:", h_pool_conv1.get_shape()
    do_fc1 = tf.nn.dropout(h_pool_conv1, .1)
    print "do_fc1.shape:", do_fc1.get_shape()
    norm1 = norm(do_fc1, 4)
    print "norm1.shape:", norm1.get_shape()

########################
#Layer 2 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-2'):
    conv2 = conv2d(norm1, weights['wc1'], biases['bc1'])
    print "conv2.shape:", conv2.get_shape()
    conv2a = conv2d(conv2, weights['wc1b'], biases['bc2'])
    print "conv3.shape:", conv2a.get_shape()
    norm2 = norm(conv2a, 4)
    print "norm1.shape:", norm2.get_shape()
    h_pool_conv2 = maxpool2d(norm2, k=2)
    print "h_pool_conv2.shape:", h_pool_conv2.get_shape()
    do_fc2 = tf.nn.dropout(h_pool_conv2, .2)
    print "do_fc1.shape:", do_fc2.get_shape()
    

########################
#Layer 3 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-3'):
    conv3 = conv2d(do_fc2, weights['wc2'], biases['bc3'])
    print "conv2.shape:", conv2.get_shape()
    conv3a = conv2d(conv3, weights['wc2b'], biases['bc4'])
    print "conv3.shape:", conv2.get_shape()
    norm3 = norm(conv3a, 4)
    print "norm3.shape:", norm3.get_shape()
    h_pool_conv3 = maxpool2d(norm3, k=2)
    print "h_pool_conv3.shape:", h_pool_conv3.get_shape()
    do_fc3 = tf.nn.dropout(h_pool_conv3, .3)
    print "do_fc3.shape:", do_fc3.get_shape()
    

########################
#Layer 4 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-4'):
    conv4 = conv2d(do_fc3, weights['wc3'], biases['bd1'])
    print "conv4.shape:", conv4.get_shape()
    conv4a = conv2d(conv4, weights['wc3b'], biases['bd2'])
    print "conv4a.shape:", conv4a.get_shape()
    norm4 = norm(conv4a, 4)
    print "norm4.shape:", norm4.get_shape()
    h_pool_conv4 = maxpool2d(norm4, k=2)
    print "h_pool_conv2.shape:", h_pool_conv4.get_shape()
    do_fc4 = tf.nn.dropout(h_pool_conv4, .4)
    print "do_fc4.shape:", do_fc4.get_shape()
   
########################
#Layer 5 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-5'):
    conv5 = conv2d(do_fc4, weights['wc4'], biases['bd2'])
    print "conv5.shape:", conv5.get_shape()
    conv5a = conv2d(conv5, weights['wc4b'], biases['bd3'])
    print "conv5a.shape:", conv5a.get_shape()
    norm5 = norm(conv5a, 4)
    print "norm5.shape:", norm5.get_shape()
    h_pool_conv5= maxpool2d(norm5, k=2)
    print "h_pool_conv5.shape:", h_pool_conv5.get_shape()
    do_fc5 = tf.nn.dropout(h_pool_conv5, .5)
    print "do_fc5.shape:", do_fc5.get_shape()
    pool5Shape = do_fc5.get_shape().as_list()

########################
#Layer 3 FC
########################
with tf.name_scope('fc-layer-1'):
    flat_fc1 = tf.reshape(do_fc5, [-1, pool5Shape[1]*pool5Shape[2]*pool5Shape[3] ])#weights['wd1'].get_shape().as_list()[0]]) #3136
    print "flat_fc1.shape:", flat_fc1.get_shape()
    fc1 = tf.add(tf.matmul(flat_fc1, weights['wd2']), biases['bd4'])
    print "fc1.shape:", fc1.get_shape()
    relu_fc1 = tf.nn.relu(fc1)
    print "relu_fc1.shape:", relu_fc1.get_shape()
    #do_fc1 = tf.nn.dropout(relu_fc1, dropout)

#with tf.name_scope('fc-layer-2'):
#    #flat_fc2 = tf.reshape(relu_fc1, weights['wd2'].get_shape().as_list()[0]) #3136
#    #   print "flat_fc2.shape:", flat_fc1.get_shape()
#    fc2 = tf.add(tf.matmul(relu_fc1, weights['wd2']), biases['bd2'])
#    print "fc2.shape:", fc2.get_shape()
#    relu_fc2 = tf.identity(fc2)
#    print "relu_fc2.shape:", relu_fc2.get_shape()
#    #do_fc2 = tf.nn.dropout(relu_fc2, dropout)

with tf.name_scope('output'):
    print "do_fc1.shape:", relu_fc1.get_shape()
    outputs = tf.add(tf.matmul(relu_fc1, weights['out']), biases['out'])
    print "outputs.shape:", outputs.get_shape()

    
with tf.name_scope('error'):
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32))

tf.summary.scalar('error_result', error)
tf.summary.scalar('accuracy_result', accuracy)
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
directory = dataset+'_'+str(num_epoch)+'_'+nonlinear_arr+'_'+str(timestamp)
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], directory)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver()


train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
start_run_time = time.time()
step = 0
for e in range(num_epoch):
    start_time = time.time()
    for b, (input_batch, target_batch) in enumerate(train_data):
        # do train step with current batch
        _, summary, batch_error, batch_acc = sess.run(
            [train_step, summary_op, error, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch, keep_prob: dropout})
        # add symmary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1

    #time    
    epoch_time = time.time()-start_time
    start_time = time.time()
    # normalise running means by number of batches
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_inputs, targets: valid_targets})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} step={3:.2f} time={4:.2f}s'
          .format(e + 1, train_error[e], train_accuracy[e], step, epoch_time))
    print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
          .format(valid_error[e], valid_accuracy[e]))

total_run_time = time.time() - start_run_time
print('Total time ={0:.2f}s'.format(total_run_time))
print('End of epoch {0:02d}: err(test)={1:.4f} acc(test)={2:.4f}'
                          .format(lowest_epoch + 1, test_error, test_accuracy))

# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()

# save run stats to a .npz file
np.savez_compressed(
    os.path.join(exp_dir, 'run.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy
)


