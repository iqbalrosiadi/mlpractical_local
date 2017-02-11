import tensorflow as tf
import os
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider as data_providers
from mlp.local_foo import get_err_and_acc, fully_connected_layer

train_data = CIFAR10DataProvider('train', batch_size=50)
valid_data = CIFAR10DataProvider('valid', batch_size=50)
learning_rate = 0.1
num_epoch = 5
num_hidden = 200


graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
        targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
    with tf.name_scope('fc-layer-1'):
        hidden_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], num_hidden)
    with tf.name_scope('output-layer'):
        outputs = fully_connected_layer(hidden_1, num_hidden, train_data.num_classes, tf.identity)
    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer().minimize(error)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32))
    tf.summary.scalar('error_result', error)
    tf.summary.scalar('accuracy_result', accuracy)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()



timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'train'), graph=graph)
valid_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'valid'), graph=graph)


sess = tf.InteractiveSession(graph=graph)
valid_inputs = valid_data.inputs
valid_targets = valid_data.to_one_of_k(valid_data.targets)
sess.run(init)
for e in range(num_epoch):
    for b, (input_batch, target_batch) in enumerate(train_data):
        _, summary = sess.run(
            [train_step, summary_op],
            feed_dict={inputs: input_batch, targets: target_batch})
        train_writer.add_summary(summary, e * train_data.num_batches + b)
        if (e + 1) % 5 == 0:
            valid_summary = sess.run(
                summary_op, feed_dict={inputs: valid_inputs, targets: valid_targets})
            valid_writer.add_summary(valid_summary, e * train_data.num_batches + b)

default_graph = tf.get_default_graph()
print('Number of operations in graph: {0}'
      .format(len(default_graph.get_operations())))



print('learning Rate: {0} and Num of Epoch: {1}'
		.format(learning_rate,num_epoch))
print('Train data: Error={0:.5f} Accuracy={1:.5f}'
      .format(*get_err_and_acc(train_data, sess, error, accuracy, inputs, targets)))
print('Valid data: Error={0:.5f} Accuracy={1:.5f}'
      .format(*get_err_and_acc(valid_data, sess, error, accuracy, inputs, targets)))

