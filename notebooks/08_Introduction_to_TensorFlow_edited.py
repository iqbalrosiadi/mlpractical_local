
# coding: utf-8

# # Introduction to TensorFlow
# 
# ## Computation graphs
# 
# In the first semester we used the NumPy-based `mlp` Python package to illustrate the concepts involved in automatically propagating gradients through multiple-layer neural network models. We also looked at how to use these calculated derivatives to do gradient-descent based training of models in supervised learning tasks such as classification and regression.
# 
# A key theme in the first semester's work was the idea of defining models in a modular fashion. There we considered  models composed of a sequence of *layer* modules, the output of each of which fed into the input of the next in the sequence and each applying a transformation to map inputs to outputs. By defining a standard interface to layer objects with each defining a `fprop` method to *forward propagate* inputs to outputs, and a `bprop` method to *back propagate* gradients with respect to the output of the layer to gradients with respect to the input of the layer, the layer modules could be composed together arbitarily and activations and gradients forward and back propagated through the whole stack respectively.
# 
# <div style='margin: auto; text-align: center; padding-top: 1em;'>
#   <img style='margin-bottom: 1em;' src='res/pipeline-graph.png' width='30%' />
#   <i>'Pipeline' model composed of sequence of single input, single output layer modules</i>
# </div>
# 
# By construction a layer was defined as an object with a single array input and single array output. This is a natural fit for the architectures of standard feedforward networks which can be thought of a single pipeline of transformations from user provided input data to predicted outputs as illustrated in the figure above. 
# 
# <div style='margin: auto; text-align: center; padding-top: 1em;'>
#   <img style='display: inline-block; padding-right: 2em; margin-bottom: 1em;' src='res/rnn-graph.png' width='30%' />
#   <img style='display: inline-block; padding-left: 2em; margin-bottom: 1em;' src='res/skip-connection-graph.png' width='30%' /> <br />
#   <i>Models which fit less well into pipeline structure: left, a sequence-to-sequence recurrent network; right, a feed forward network with skip connections.</i>
# </div>
# 
# Towards the end of last semester however we encountered several models which do not fit so well in to this pipeline like structure. For instance (unrolled) recurrent neural networks tend to have inputs feeding in to and outputs feeding out from multiple points along a deep feedforward model corresponding to the updates of the hidden recurrent state, as illustrated in the left panel in the figure above. It is not trivial to see how to map this structure to our layer based pipeline. Similarly models with skip connections between layers as illustrated in the right panel of the above figure also do not fit particularly well in to a pipeline structure.
# 
# Ideally we would like to be able to compose modular components in more general structures than the pipeline structure we have being using so far. In particular it turns out to be useful to be able to deal with models which have structures defined by arbitrary [*directed acyclic graphs*](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs), that is graphs connected by directed edges and without any directed cycles. Both the recurrent network and skip-connections examples can be naturally expressed as DAGs as well many other model structures.
# 
# When working with these more general graphical structures, rather than considering a graph made up of layer modules, it often more useful to consider lower level mathematical operations or *ops* that make up the computation as the fundamental building block. A DAG composed of ops is often termed a *computation graph*. Those who sat MLPR last semester will have [come across this terminology already](http://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w5a_backprop.html). The backpropagation rules we used to propagate gradients through a stack of layer modules can be naturally generalised to apply to computation graphs, with this method of applying the chain rule to automatically propagate gradients backwards through a general computation graph also sometimes termed [*reverse-mode automatic differentiation*](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation).
# 
# <div style='margin: auto; text-align: center; padding-top: 1em;'>
#   <img style='margin-bottom: 1em;' src='res/affine-transform-graph.png' width='40%' />
#   <i>Computation / data flow graph for an affine transformation $\boldsymbol{y} = \mathbf{W}\boldsymbol{x} + \boldsymbol{b}$</i>
# </div>
# 
# The figure above shows a very simple computation graph corresponding to the mathematical expression  $\boldsymbol{y} = \mathbf{W}\boldsymbol{x} + \boldsymbol{b}$, i.e. the affine transformation we encountered last semester. Here the nodes of the graph are operations and the edges the vector or matrix values passed between operations. The opposite convention with nodes as values and edges as operations is also sometimes used. Note that just like there was ambiguity about what to define as a layer (as discussed previously at beginning of the [third lab notebook](03_Multiple_layer_models.ipynb), there are a range of choices for the level of abstraction to use in the op nodes in a computational graph. For instance we could also have chosen to express the above computational graph with a single `AffineTransform` op node with three inputs (one matrix, two vector) and one vector output. Equally we might choose to express the `MatMul` op in terms of the underlying individual scalar addition and multiplication operations. What to consider an operation is therefore somewhat a matter of choice and what is convenient in a particular setting.
# 
# ##  TensorFlow
# 
# To allow us to work with models defined by more general computation graphs and to avoid the need to write `fprop` and `bprop` methods for each new model component we want to try out, this semester we will be using the open-source computation graph framework [TensorFlow](https://www.tensorflow.org/), originally developed by the Google Brain team:
# 
# > TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs 
# in a desktop, server, or mobile device with a single API.
# 
# TensorFlow allows complex computation graphs (also known as data flow graphs in TensorFlow parlance) to be defined via a Python interface, with efficient C++ implementations for running the corresponding operations on different devices. TensorFlow also includes tools for automatic gradient computation and a large and growing suite of pre-define operations useful for gradient-based training of machine learning models.
# 
# In this notebook we will introduce some of the basic elements of constructing, training and evaluating models with TensorFlow. This will use similar material to some of the [official TensorFlow tutorials](https://www.tensorflow.org/tutorials/) but with an additional emphasis of making links to the material covered in this course last semester. For those who have not used a computational graph framework such as TensorFlow or Theano before you may find the [basic usage tutorial](https://www.tensorflow.org/get_started/basic_usage) useful to go through.
# 
# ### Installing TensorFlow
# 
# To install TensorFlow, open a terminal, activate your Conda `mlp` environment using
# 
# ```
# source activate mlp
# ```
# 
# and then run
# 
# ```
# conda install -c conda-forge tensorflow=0.12.1
# ```
# 
# This should locally install version 0.12.1 (the latest stable version as of the time of writing) of TensorFlow in to your Conda environment. After installing TensorFlow you may need to restart the kernel in the notebook to allow it to imported.

# ## Exercise 1: MNIST softmax regression
# 
# As a first example we will train a simple softmax regression model to classify handwritten digit images from the MNIST data set encountered last semester (for those fed up of working with MNIST: don't worry you will soon be moving on to other datasets!). This is equivalent to the model implemented in the first exercise of the third lab notebook. We will walkthrough constructing an equivalent model in TensorFlow and explain new TensorFlow model concepts as we use them. You should run each cell as you progress through the exercise.
# 
# Similarly to the common convention of importing NumPy under the shortform alias `np` it is common to import the Python TensorFlow top-level module under the alias `tf`.

# In[ ]:

import tensorflow as tf


# We begin by defining [*placeholder*](https://www.tensorflow.org/api_docs/python/io_ops/placeholders) objects for the data inputs and targets arrays. These are nodes in the computation graph which we will later *feed* in external data to e.g. batches of training set inputs and targets during each training set. This abstraction allows us to reuse the same computation graph for different data inputs - we can think of placeholders as acting equivalently to the arguments of a function. It is actually possible to feed data into any node in a TensorFlow graph however the advantage of using a placeholder is that is *must* always have a value fed into it (an exception will be raised if a value isn't provided) and no arbitrary alternative values needs to be entered.
# 
# The `tf.placeholder` function has three arguments:
# 
#   * `dtype` : The [TensorFlow datatype](https://www.tensorflow.org/api_docs/python/framework/tensor_types) for the tensor e.g. `tf.float32` for single-precision floating point values.
#   * `shape` (optional) : An iterable defining the shape (size of each dimension) of the tensor e.g. `shape=(5, 2)` would indicate a 2D tensor ($\sim$ matrix) with first dimension of size 5 and second of size 2. An entry of `None` in the shape definition corresponds to the corresponding dimension size being left unspecified, so for example `shape=(None, 28, 28)` would allow any 3D inputs with final two dimensions of size 28 to be inputted.
#   * `name` (optional): String argument defining a name for the tensor which can be useful when visualising a computation graph and for debugging purposes.
#   
# As we will generally be working with batches of datapoints, both the `inputs` and `targets` will be 2D tensors with first dimension corresponding to the batch size (set as `None` here to allow it to specified later) and second dimension the size of each input or output vector. As in the previous semester's work we will use a 1-of-$k$ encoding for the class targets so each output corresponds to a vector of length 10 (number of digit classes).

# In[ ]:

inputs = tf.placeholder(tf.float32, [None, 784], 'inputs')
targets = tf.placeholder(tf.float32, [None, 10], 'targets')


# We now define [*variable*](https://www.tensorflow.org/api_docs/python/state_ops/variables) objects for the model parameters. Variables are stateful tensors in the computation graph - they have to be explicitly initialised and their internal values can be updated as part of the operations in a graph e.g. gradient updates to model parameter during training. They can also be saved to disk and pre-saved values restored in to a graph at a later time.
# 
# The `tf.Variable` constructor takes an `initial_value` as its first argument; this should be a TensorFlow tensor which specifies the initial value to assign to the variable, often a constant tensor such as all zeros, or random samples from a distribution.

# In[ ]:

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))


# We now build the computation graph corresponding to producing the predicted outputs of the model (log unnormalised class probabilities) given the data inputs and model parameters. We use the TensorFlow [`matmul`](https://www.tensorflow.org/api_docs/python/math_ops/matrix_math_functions#matmul) op to compute the matrix-matrix product between the 2D array of input vectors and the weight matrix parameter variable. TensorFlow [overloads all of the common arithmetic operators](http://stackoverflow.com/a/35095052) for tensor objects so `x + y` where at least one of `x` or `y` is a tensor instance (both `tf.placeholder` and `tf.Variable` return (sub-classes) of `tf.Tensor`) corresponds to the TensorFlow elementwise addition op `tf.add`. Further elementwise binary arithmetic operators like addition follow NumPy style [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), so in the expression below the `+ biases` sub-expression will correspond to creating an operation in the computation graph which adds the biases vector to each of the rows of the 2D tensor output of the `matmul` op.

# In[ ]:

outputs = tf.matmul(inputs, weights) + biases


# While we could have defined `outputs` as the softmax of the expression above to produce normalised class probabilities as the outputs of the model, as discussed last semester when using a softmax output combined with a cross-entropy error function it usually desirable from a numerical stability and efficiency perspective to wrap the softmax computation in to the error computation (as done in the `CrossEntropySoftmaxError` class in our `mlp` framework). 
# 
# In TensorFlow this can be achieved with the `softmax_cross_entropy_with_logits` op which is part of the `tf.nn` submodule which contains a number of ops specifically for neural network type models. This op takes as its first input log unnormalised class probabilities (sometimes termed logits) and as second input the class label targets which should be of the same dimension as the first input. By default the last dimension of the input tensors is assumed to correspond to the class dimension - this can be altered via an optional `dim` argument.
# 
# The output of the `softmax_cross_entropy_with_logits` op here is a 1D tensor with a cross-entropy error value for each data point in the batch. We wish to minimise the mean cross-entropy error across the full dataset and will use the mean of the error on the batch as a stochastic estimator of this value. In TensorFlow ops which *reduce* a tensor along a dimension(s) by for example taking a sum, mean or product are prefixed with `reduce`, with the default behaviour being to perform the reduction across all dimensions of the input tensor and return a scalar output. Therefore the second line below will take the per data point cross-entropy errors and produce a single mean value across the whole batch.

# In[ ]:

per_datapoint_errors = tf.nn.softmax_cross_entropy_with_logits(outputs, targets)
error = tf.reduce_mean(per_datapoint_errors)


# Although for the purposes of training we will use the cross-entropy error as this is differentiable, for evaluation we will also be interested in the classification accuracy i.e. what proportion of all of the predicted classes correspond to the true target label. We can calculate this in TensorFlow similarly to how we used NumPy to do this previously - we use the TensorFlow `tf.argmax` op to find the index of along the class dimension corresponding to the maximum predicted class probability and check if this is equal to the index along the class dimension of the 1-of-$k$ encoded target labels. Analagously to the error computation above, this computes per-datapoint values which we then need to average across with a `reduce_mean` op to produce the classification accuracy for a batch.

# In[ ]:

per_datapoint_pred_is_correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(per_datapoint_pred_is_correct, tf.float32))


# As mentioned previously TensorFlow is able to automatically calculate gradients of scalar computation graph outputs with respect to tensors in the computation graph. We can explicitly construct a new sub-graph corresponding to the gradient of a scalar with respect to one or more tensors in the graph using the [`tf.gradients`](https://www.tensorflow.org/api_docs/python/train/gradient_computation) function. 
# 
# TensorFlow also however includes a number of higher-level `Optimizer` classes in the `tf.train` module that internally deal with constructing graphs corresponding to the gradients of some scalar loss with respect to one or more `Variable` tensors in the graph (usually corresponding to model parameters) and then using these gradients to update the variables (roughly equivalent to the `LearningRule` classes in the `mlp` framework). The most basic `Optimizer` instance is the `GradientDescentOptimizer` which simply adds operations corresponding to basic (stochastic) gradient descent to the graph (i.e. no momentum, adaptive learning rates etc.). The `__init__` constructor method for this class takes one argument `learning_rate` corresponding to the gradient descent learning rate / step size encountered previously.
# 
# Usually we are not interested in the `Optimizer` object other than in adding operations in the graph corresponding to the optimisation steps. This can be achieved using the `minimize` method of the object which takes as first argument the tensor object corresponding to the scalar loss / error to be minimized. A further optional keyword argument `var_list` can be used to specify a list of variables to compute the gradients of the loss with respect to and update; by default this is set to `None` which indicates to use all trainable variables in the current graph. The `minimize` method returns an operation corresponding to applying the gradient updates to the variables - we need to store a reference to this to allow us to run these operations later. Note we do not need to store a reference to the optimizer as we have no further need of this object hence commonly the steps of constructing the `Optimizer` and calling `minimize` are commonly all applied in a single line as below.

# In[ ]:

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(error)


# We have now constructed a computation graph which can compute predicted outputs, use these to calculate an error value (and accuracy) and use the gradients of the error with respect to the model parameter variables to update their values with a gradient descent step.
# 
# Although we have defined our computation graph, we have not yet initialised any tensor data in memory - all of the tensor variables defined above are just symbolic representations of parts of the computation graph. We can think of the computation graph as a whole as being similar to a function - it defines a sequence of operations but does not directly run those operations on data itself.
# 
# To run the operations in (part of) a TensorFlow graph we need to create a [`Session`](https://www.tensorflow.org/api_docs/python/client/session_management) object:
# 
# > A `Session` object encapsulates the environment in which `Operation` objects are executed, and `Tensor` objects are evaluated.
# 
# A session object can be constructed using either `tf.Session()` or `tf.InteractiveSession()`. The only difference in the latter is that it installs itself as the default session on construction. This can be useful in interactive contexts such as shells or the notebook interface as an alternative to running a graph operation using the session `run` method (see below) is to call the `eval` method of an operation e.g. `op.eval()`; generally a session for the op to be run in needs to be passed to `eval` however if an interactive session is used this is set as a default to use in `eval` calls.

# In[ ]:

sess = tf.InteractiveSession()


# The key property of a session object is its `run` method. This takes an operation (or list of operations) in a defined graph as an argument and runs the parts of the computation graph necessary to evaluate the output(s) (if any) of the operation(s), and additionally performs any updates to variables states defined by the graph (e.g. gradient updates of parameters). The output values if any of the operation(s) are returned by the `run` call.
# 
# A standard operation which needs to be called before any other operations on a graph which includes variable nodes is a variable *initializer* operation. This, as the name suggests, initialises the values of the variables in the session to the values defined by the `initial_value` argument when adding the variables to the graph. For instance for the graph we have defined here this will initialise the `weights` variable value in the session to a 2D array of zeros of shape `(784, 10)` and the `biases` variable to a 1D array of shape `(10,)`.
# 
# We can access initializer ops for each variable individually using the `initializer` property of the variables in question and then individually run these, however a common pattern is to use the `tf.global_variables_initializer()` function to create a single initializer op which will initialise all globally defined variables in the default graph and then run this as done below.

# In[ ]:

init_op = tf.global_variables_initializer()
sess.run(init_op)


# We are now almost ready to begin training our defined model, however as a final step we need to create objects for accessing batches of MNIST input and target data. In the tutorial code provided in `tf.examples.tutorials.mnist` there is an `input_data` sub-module which provides a `read_data_sets` function for downloading the MNIST data and constructing an object for iterating over MNIST data. However in the `mlp` package we already have the MNIST data provider class we used extensively last semester and a corresponding local copy of the MNIST data so we will use that here as it provides all the necessary functionality.

# In[ ]:

import mlp.data_providers as data_providers
train_data = data_providers.MNISTDataProvider('train', batch_size=50)
valid_data = data_providers.MNISTDataProvider('valid', batch_size=50)


# We are now all set to train our model. As when training models last semester, the training procedure will involve two nested loops - an outer loop corresponding to multiple full-passes through the dataset or *epochs* and an inner loop iterating over individual batches in the training data.
# 
# The `init_op` we ran with `sess.run` previously did not depend on the placeholders `inputs` and `target` in our graph, so we simply ran it with `sess.run(init_op)`. The `train_step` operation corresponding to the gradient based updates of the `weights` and `biases` parameter variables does however depend on the `inputs` and `targets` placeholders and so we need to specify values to *feed* into these placeholders; as we wish the gradient updates to be calculated using the gradients with respect to a batch of inputs and targets, the values that we feed in are the input and target batches. This is specified using the keyword `feed_dict` argument to the session `run` method. As the name suggests this should be a Python dictionary (`dict`) with keys corresponding to references to the tensors in the graph to feed values in to and values the corresponding array values to feed in (typically NumPy `ndarray` instances) - here we have `feed_dict = {inputs: input_batch, targets: target_batch}`.
# 
# Another difference in our use of the session `run` method below is that we call it with a list of two operations - `[train_step, error]` rather than just a single operation. This allows the output (and variable updates) of multiple operations in a graph to be evaluated together - here we both run the `train_step` operation to update the parameter values and evaluate the `error` operation to return the mean error on the batch. Although we could split this into two separate session `run` calls, as the operations calculating the batch error will need to be evaluated when running the `train_step` operation (as this is the value gradients are calculated with respect to) this would involve redoing some of the computation and so be less efficient than combining them in a single `run` call.
# 
# As we are running two different operations, the `run` method returns two values here. The `train_step` operation has no outputs and so the first return value is `None` - in the code below we assign this to `_`, this being a common convention in Python code for assigning return values we are not interested in using. The second return value is the average error across the batch which we assign to `batch_error` and use to keep a running average of the dataset error across the epochs.

# In[ ]:

num_epoch = 5
for e in range(num_epoch):
    running_error = 0.
    for input_batch, target_batch in train_data:
        _, batch_error = sess.run(
            [train_step, error], 
            feed_dict={inputs: input_batch, targets: target_batch})
        running_error += batch_error
    running_error /= train_data.num_batches
    print('End of epoch {0}: running error average = {1:.2f}'.format(e + 1, running_error))


# To check your understanding of using sessions objects to evaluate parts of a graph and feeding values in to a graph, complete the definition of the function in the cell below. This should iterate across all batches in a provided data provider and calculate the error and classification accuracy for each, accumulating the average error and accuracy values across the whole dataset and returning these as a tuple.

# In[ ]:

def get_error_and_accuracy(data):
    """Calculate average error and classification accuracy across a dataset.
    
    Args:
        data: Data provider which iterates over input-target batches in dataset.
        
    Returns:
        Tuple with first element scalar value corresponding to average error
        across all batches in dataset and second value corresponding to
        average classification accuracy across all batches in dataset.
    """
    err = 0
    acc = 0
    for input_batch, target_batch in data:
        err += sess.run(error, feed_dict={inputs: input_batch, targets: target_batch})
        acc += sess.run(accuracy, feed_dict={inputs: input_batch, targets: target_batch})
    err /= data.num_batches
    acc /= data.num_batches
    return err, acc


# Test your implementation by running the cell below - this should print the error and accuracy of the trained model on the validation and training datasets if implemented correctly.

# In[ ]:

print('Train data: Error={0:.2f} Accuracy={1:.2f}'
      .format(*get_error_and_accuracy(train_data)))
print('Valid data: Error={0:.2f} Accuracy={1:.2f}'
      .format(*get_error_and_accuracy(valid_data)))


# ## Exercise 2: Explicit graphs, name scopes, summaries and TensorBoard
# 
# In the exercise above we introduced most of the basic concepts needed for constructing graphs in TensorFlow and running graph operations. In an attempt to avoid introducing too many new terms and syntax at once however we skipped over some of the non-essential elements of creating and running models in TensorFlow, in particular some of the provided functionality for organising and structuring the computation graphs created and for monitoring the progress of training runs.
# 
# Now that you are hopefully more familiar with the basics of TensorFlow we will introduce some of these features as they are likely to provide useful when you are building and training more complex models in the rest of this semester.
# 
# Although we started off by motivating TensorFlow as a framework which builds computation graphs, in the code above we never explicitly referenced a graph object. This is because TensorFlow always registers a default graph at start up and all operations are added to this graph by default. The default graph can be accessed using `tf.get_default_graph()`. For example running the code in the cell below will assign a reference to the default graph to `default_graph` and print the total number of operations in the current graph definition.

# In[ ]:

default_graph = tf.get_default_graph()
print('Number of operations in graph: {0}'
      .format(len(default_graph.get_operations())))


# We can also explicitly create a new graph object using `tf.Graph()`. This may be useful if we wish to build up several independent computation graphs.

# In[ ]:

graph = tf.Graph()


# To add operations to a constructed graph object, we use the `graph.as_default()` [context manager](http://book.pythontips.com/en/latest/context_managers.html). Context managers are used with the `with` statement in Python - `with context_manager:` opens a block in Python in which a special `__enter__` method of the `context_manager` object is called before the code in the block is run and a further special `__exit__` method is run after the block code has finished execution. This can be used to for example manage allocation of resources (e.g. file handles) but also to locally change some 'context' in the code, e.g. in the example here `graph.as_default()` is a context manager which changes the default graph within the following block to be `graph` before returning to the previous default graph once the block code is finished running. Context managers are used extensively in TensorFlow so it is worth being familiar with how they work.
# 
# Another common context manager usage in TensorFlow is to define *name scopes*. As we encountered earlier individual operations in a TensorFlow graph can be assigned names. As we will see later this is useful for making graphs interpretable when we use the tools provided in TensorFlow for visualising them. As computation graphs can become very big (even the quite simple graph we created in the first exercise has around 100 operations in it) even with interpretable names attached to the graph operations it can still be difficult to understand and debug what is happening in a graph. Therefore rather than simply allowing a single-level naming scheme to be applied to the individual operations in the graph, TensorFlow supports hierachical naming of sub-graphs. This allows sets of related operations to be grouped together under a common name, and thus allows both higher and lower level structure in a graph to be easily identified.
# 
# This hierarchical naming is performed by using the name scope context manager `tf.name_scope('name')`. Starting a block `with tf.name_scope('name'):`, will cause all the of the operations added to a graph within that block to be grouped under the name specified in the `tf.name_scope` call. Name scope blocks can be nested to allow finer-grained sub-groupings of operations. Name scopes can be used to group operations at various levels e.g. operations corresponding to inference/prediction versus training, grouping operations which correspond to the classical definition of a neural network layer etc.
# 
# The code in the cell below uses both a `graph.as_default()` context manager and name scopes to create a second copy of the computation graph corresponding to softmax regression that we constructed in the previous exercise.

# In[ ]:

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
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(error)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32))


# As hinted earlier TensorFlow comes with tools for visualising computation graphs. In particular [TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) is an interactive web application for amongst other things visualising TensorFlow computation graphs (we will explore some of its other functionality in the latter part of the exercise). Typically TensorBoard in launched from a terminal and a browser used to connect to the resulting locally running TensorBoard server instance. However for the purposes of graph visualisation it is also possible to embed a remotely-served TensorBoard graph visualisation interface in a Jupyter notebook using the helper function below (a slight variant of the recipe in [this notebook](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)).
# 
# <span style='color: red; font-weight: bold;'>Note: The code below seems to not work for some people when accessing the notebook in Firefox. You can either try loading the notebook in an alternative browser, or just skip this section for now and explore the graph visualisation tool when launching TensorBoard below.</span>

# In[ ]:

from IPython.display import display, HTML
import datetime

def show_graph(graph_def, frame_size=(900, 600)):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(height=frame_size[1], data=repr(str(graph_def)), id='graph'+timestamp)
    iframe = """
        <iframe seamless style="width:{width}px;height:{height}px;border:0" srcdoc="{src}"></iframe>
    """.format(width=frame_size[0], height=frame_size[1] + 20, src=code.replace('"', '&quot;'))
    display(HTML(iframe))


# Run the cell below to display a visualisation of the graph we just defined. Notice that by default all operations within a particular defined name scope are grouped under a single node; this allows the top-level structure of the graph and how data flows between the various components to be easily visualised. We can also expand these nodes however to interrogate the operations within them - simply double-click on one of the nodes to do this (double-clicking on the expanded node will cause it to collapse again). If you expand the `model` node you should see a graph closely mirroring the affine transform example given as a motivation above.

# In[ ]:

show_graph(graph)


# To highlight how using name scopes can be very helpful in making these graph visualisations more interpretable, running the cell below will create a corresponding visualisation for the graph created in the first exercise, which contains the same operations but without the name scope groupings.

# In[ ]:

show_graph(tf.get_default_graph())


# A common problem when doing gradient based training of complex models is how to monitor progress during training. In the `mlp` framework we used last semester we included some basic logging functionality for recording training statistics such as training and validation set error and classificaton accuracy at the end of each epoch. By printing the log output this allowed basic monitoring of how training was proceeding. However due to the noisiness of the the training procedures the raw values printed were often difficult to interpret. After a training run we often plotted training curves to allow better visualisation of how the run went but this could only be done after a run was completed and required a lot of boilerplate code to be written (or copied and pasted...).
# 
# TensorFlow [*summary* operations](https://www.tensorflow.org/api_docs/python/summary/) are designed to help deal with this issue. Summary operations can be added to the graph to allow summary statistics to be computed and serialized to event files. These event files can then be loaded in TensorBoard *during training* to allow continuous graphing of for example the training and validation set error during training. As well as summary operations for monitoring [scalar](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#scalar) values such as errors or accuracies, TensorFlow also includes summary operations for monitoring [histograms](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#histogram) of tensor quanties (e.g. the distribution of a set of weight parameters), displaying [images](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#image) (for example for checking if random augmentations being applied to image inputs are producing reasonable outputs) and even playing back [audio](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#audio).
# 
# The cell below adds two simple scalar summary operations to our new graph for monitoring the error and classification accuracy. While we can keep references to all of the summary ops we add to a graph and make sure to run them all individually in the session during training, as with variable initialisation, TensorFlow provides a convenience method to avoid having to write a lot of boilerplate code like this. The `tf.summary.merge_all()` function returns an merged op corresponding to all of the summary ops that have been added to the current default graph. We can then just run this one merged op in our session to generate all the summaries we have added.

# In[ ]:

with graph.as_default():
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()


# In addition to the (merged) summary operation, we also need to define a *summary writer* object(s) to specify where the summaries should be written to on disk. The `tf.summary.FileWriter` class constructor takes a `logdir` as its first argument which should specify the path to a directory where event files should be written to. In the code below the log directory is specified as a local directory `tf-log` plus a timestamp based sub-directory within this to keep event files corresponding to different runs separated. The `FileWriter` constructor also accepts an optional `graph` argument which we here set to the graph we just populated with summaries. We construct separate writer objects for summaries on the training and validation datasets.

# In[ ]:

import os
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'train'), graph=graph)
valid_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'valid'), graph=graph)


# The final step in using summaries is to run the merged summary op at the appropriate points in training and to add the outputs of the run summary operations to the writers. Here we evaluate the summary op on each training dataset batch and after every 100th batch evaluate the summary op on the whole validation dataset, writing the outputs of each to the relevant writers.
# 
# If you run the cell below, you should be able to visualise the resulting training run summaries by launching TensorBoard within a shell with
# 
# ```bash
# tensorboard --logdir=[path/to/tf-log]
# ```
# 
# where `[path/to/tf-log]` is replaced with the path to the `tf-log` directory specified abovem and then opening the URL `localhost:6006` in a browser.

# In[ ]:

with graph.as_default():
    init = tf.global_variables_initializer()
sess = tf.InteractiveSession(graph=graph)
num_epoch = 5
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


# That completes our basic introduction to TensorFlow. If you want more to explore more of TensorFlow before beginning your project for this semester, you may wish to go through some of the [official tutorials](https://www.tensorflow.org/tutorials/) or some of the many sites with unofficial tutorials e.g. the series of notebooks [here](https://github.com/aymericdamien/TensorFlow-Examples). If you have time you may also wish to have a go at the optional exercise below.

# ## Optional exercise: multiple layer MNIST classifier using `contrib` modules
# 
# As well as the core officially supported codebase, TensorFlow is distributed with a series of contributed modules under [`tensorflow.contrib`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib). These tend to provide higher level interfaces for constructing and running common forms of computational graphs which can allow models to be constructed with much more concise code. The interfaces of the `contrib` modules tend to be (even) less stable than the core TensorFlow Python interface and they are also more restricted in the sorts of models that can be created. Therefore it is worthwhile to also be familiar with constructing models with the operations available in the core TensorFlow codebase; you can also often mix and match use of 'native' TensorFlow and functions from `contrib` modules.
# 
# As an optional extension exercise, construct a deep MNIST classifier model, either using TensorFlow operations directly as above or using one (or more) of the higher level interfaces defined in `contrib` modules such as [`tensorflow.contrib.learn`](https://www.tensorflow.org/tutorials/tflearn/), [`tensorflow.contrib.layers`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/layers) or [`tensorflow.contrib.slim`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim). You should choose an appropriate model architecture (number and width of layers) and choice of activation function based on your experience fitting models from last semester.
# 
# As well as exploring the use of the interfaces in `contrib` modules you may wish to explore the more advanced optimizers available in [`tensorflow.train`](https://www.tensorflow.org/versions/r0.11/api_docs/python/train) such as [`tensorflow.train.AdamOptimizer`](https://www.tensorflow.org/versions/r0.11/api_docs/python/train/optimizers#AdamOptimizer) and [`tensorflow.train.AdagradOptimizer`](https://www.tensorflow.org/versions/r0.11/api_docs/python/train/optimizers#AdagradOptimizer) corresponding to the adaptive learning rules implemented in the first coursework last semester.

# In[ ]:



