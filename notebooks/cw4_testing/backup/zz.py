#!/disk/scratch/mlp/miniconda2/bin/python
import os
import sys
import datetime
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
num_epoch = 1
dropout = 0.5 # Dropout, probability to keep units
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
    train_data = CIFAR100DataProvider('train', batch_size=5)
    valid_data = CIFAR100DataProvider('valid', batch_size=5)
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
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=.1)),
    'wc1a': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=.1)),
    'wc2a': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=.1)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=.1)),
    'wc3a': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=.1)),
    #'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384]))
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([4*4*64, 4096])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([8192, 1024], stddev=2. / (8192 + 1024)**0.5)),
    'wd2': tf.Variable(tf.truncated_normal([2048, 1024], stddev=2. / (2048 + 1024)**0.5)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, train_data.num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
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
#Layer 3 CONV>POOL>NORM
########################
with tf.name_scope('conv-stack-3'):
    conv3 = conv2d(inputs, weights['wc1'], biases['bc1'])
    print "conv2.shape:", conv3.get_shape()
    h_pool_conv2 = maxpool2d(conv3, k=2)
    print "h_pool_conv2.shape:", h_pool_conv2.get_shape()
    pool5Shape = h_pool_conv2.get_shape().as_list()


########################
#Layer 3 FC
########################
with tf.name_scope('fc-layer-1'):
    flat_fc1 = tf.reshape(h_pool_conv2, [-1, pool5Shape[1]*pool5Shape[2]*pool5Shape[3] ])#weights['wd1'].get_shape().as_list()[0]]) #3136
    print "flat_fc1.shape:", flat_fc1.get_shape()
    fc1 = tf.add(tf.matmul(flat_fc1, weights['wd1']), biases['bd1'])
    print "fc1.shape:", fc1.get_shape()
    relu_fc1 = tf.nn.relu(fc1)
    print "relu_fc1.shape:", relu_fc1.get_shape()
    #do_fc1 = tf.nn.dropout(relu_fc1, dropout)

with tf.name_scope('output'):
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
step = 0
for e in range(num_epoch):
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
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} step={3:.2f}'
          .format(e + 1, train_error[e], train_accuracy[e], step))
    print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
          .format(valid_error[e], valid_accuracy[e]))

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


