# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc0': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([1, 1, 32, 64], stddev=.1)),
    'wc1b': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.truncated_normal([1, 1, 128, 192], stddev=.1)),
    'wc2b': tf.Variable(tf.truncated_normal([2, 2, 192, 240], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc3': tf.Variable(tf.truncated_normal([1, 1, 240, 260], stddev=.1)),
    'wc3b': tf.Variable(tf.truncated_normal([2, 2, 260, 280], stddev=.1)),
    # 5x5 conv, 1 input, 32 outputs
    'wc4': tf.Variable(tf.truncated_normal([1, 1, 280, 280], stddev=.1)),
    'wc4b': tf.Variable(tf.truncated_normal([2, 2, 280, 300], stddev=.1)),
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