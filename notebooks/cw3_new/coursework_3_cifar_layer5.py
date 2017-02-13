import tensorflow as tf
import sys
import os
from decimal import Decimal
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
from mlp.local_foo import get_err_and_acc, fully_connected_layer

nonlinear_arrs = ['tf.sigmoid', 'tf.nn.softsign', 'tf.nn.softplus']
num_epoch = Decimal(str(sys.argv[2]))
num_hidden = 200
commands = {
    'tf.nn.relu' : tf.nn.relu,
    'tf.tanh' : tf.tanh,
    'tf.nn.crelu' : tf.nn.crelu,
    'tf.nn.relu6' : tf.nn.relu6,
    'tf.sigmoid' : tf.sigmoid,
    'tf.nn.softplus' : tf.nn.softplus,
    'tf.nn.softsign' : tf.nn.softsign
}



for nonlinear_arr in nonlinear_arrs:
    graph = tf.Graph()
    if (str(sys.argv[1])=='10'):
        train_data = CIFAR10DataProvider('train', batch_size=50)
        valid_data = CIFAR10DataProvider('valid', batch_size=50)
        dataset ='C10'

    if (str(sys.argv[1])=='100'):
        train_data = CIFAR100DataProvider('train', batch_size=50)
        valid_data = CIFAR100DataProvider('valid', batch_size=50)
        dataset ='C100'

    with graph.as_default():
        with tf.name_scope('data'):
            inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
            targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
        with tf.name_scope('fc-layer-1'):
            hidden_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], num_hidden, commands[nonlinear_arr])
        with tf.name_scope('fc-layer-2'):
            hidden_2 = fully_connected_layer(hidden_1, num_hidden, num_hidden, commands[nonlinear_arr])
        with tf.name_scope('fc-layer-3'):
            hidden_3 = fully_connected_layer(hidden_2, num_hidden, num_hidden, commands[nonlinear_arr])
        with tf.name_scope('fc-layer-4'):
            hidden_4 = fully_connected_layer(hidden_3, num_hidden, num_hidden, commands[nonlinear_arr])
        with tf.name_scope('fc-layer-5'):
            hidden_5 = fully_connected_layer(hidden_4, num_hidden, num_hidden, commands[nonlinear_arr])
        with tf.name_scope('output-layer'):
            outputs = fully_connected_layer(hidden_5, num_hidden, train_data.num_classes, tf.identity)
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = dataset+'_5lay_'+str(num_epoch)+'_'+nonlinear_arr+'_'+str(timestamp)
    train_writer = tf.summary.FileWriter(os.path.join('tf-log', directory, 'train'), graph=graph)
    valid_writer = tf.summary.FileWriter(os.path.join('tf-log', directory, 'valid'), graph=graph)
    print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxooxoxoxoxxooxoxox')
    print('nonlinear array: {0} and Num of Epoch: {1}'
            .format(nonlinear_arr,num_epoch))
    print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxooxoxoxoxxooxoxox')

    sess = tf.InteractiveSession(graph=graph)
    valid_inputs = valid_data.inputs
    valid_targets = valid_data.to_one_of_k(valid_data.targets)
    #valid_targets = valid_data.target_batch
    sess.run(init)
    for e in range(num_epoch):
        running_error = 0.
        running_accuracy = 0.
        for b, (input_batch, target_batch) in enumerate(train_data):
            _, summary = sess.run(
                [train_step, summary_op],
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += sess.run(error, feed_dict={inputs: input_batch, targets: target_batch})
            running_accuracy += sess.run(accuracy, feed_dict={inputs: input_batch, targets: target_batch})
            train_writer.add_summary(summary, e * train_data.num_batches + b)
            if (b + 1) % 5 == 0:#if b % 100 == 0: #(e + 1) % 5 == 0:
                valid_summary = sess.run(
                    summary_op, feed_dict={inputs: valid_inputs, targets: valid_targets})
                valid_writer.add_summary(valid_summary, e * train_data.num_batches + b)
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.4f} acc(train)={2:.4f}'
                .format(e + 1, running_error, running_accuracy))

    print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxooxoxoxoxxooxoxox')
    default_graph = tf.get_default_graph()
    print('Number of operations in graph: {0}'
          .format(len(default_graph.get_operations())))
    print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxooxoxoxoxxooxoxox')
    print('Train data: Error={0:.5f} Accuracy={1:.5f}'
          .format(*get_err_and_acc(train_data, sess, error, accuracy, inputs, targets)))
    print('Valid data: Error={0:.5f} Accuracy={1:.5f}'
          .format(*get_err_and_acc(valid_data, sess, error, accuracy, inputs, targets)))

    sess.close()

