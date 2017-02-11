import tensorflow as tf
import os
import datetime
import mlp.data_providers as data_providers
train_data = data_providers.MNISTDataProvider('train', batch_size=50)
valid_data = data_providers.MNISTDataProvider('valid', batch_size=50)
learning_rate = 0.1
num_epoch = 5


graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
        targets = tf.placeholder(tf.float32, [None, 10], name='targets')
    with tf.name_scope('parameters'):
        weights = tf.Variable(tf.zeros([784, 10]), name='weights')
        biases = tf.Variable(tf.zeros([10]), name='biases')
    with tf.name_scope('model'):
        outputs = tf.matmul(inputs, weights) + biases
    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
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
        if b % 100 == 0:
            valid_summary = sess.run(
                summary_op, feed_dict={inputs: valid_inputs, targets: valid_targets})
            valid_writer.add_summary(valid_summary, e * train_data.num_batches + b)

default_graph = tf.get_default_graph()
print('Number of operations in graph: {0}'
      .format(len(default_graph.get_operations())))


def get_err_and_acc(data):
    err = 0
    acc = 0
    for input_batch, target_batch in data:
        err += sess.run(error, feed_dict={inputs: input_batch, targets: target_batch})
        acc += sess.run(accuracy, feed_dict={inputs: input_batch, targets: target_batch})
    err /= data.num_batches
    acc /= data.num_batches
    return err, acc

print('learning Rate: {0} and Num of Epoch: {1}'
		.format(learning_rate,num_epoch))
print('Train data: Error={0:.5f} Accuracy={1:.5f}'
      .format(*get_err_and_acc(train_data)))
print('Valid data: Error={0:.5f} Accuracy={1:.5f}'
      .format(*get_err_and_acc(valid_data)))

