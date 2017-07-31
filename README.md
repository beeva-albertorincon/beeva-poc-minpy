# minpy

### Motivation

The goal of this PoC is to check the feasibility of using Minpy instead of numpy.
In that direction, the document below tries to answer the following questions:

* What is minpy?
* What does it let you do?
* How faster is than numpy?


#### What is Minpy?

*Yep, another deep learning tool... but let explain it, seems interesting*

>It is an interface for **numpy above MXNet backend**, with many operations running on GPU.

In other words, it allows you to use both numpy and MXNet within a **single interface**.

In addition, it includes [support for TensorBoard visualization](http://minpy.readthedocs.io/en/latest/tutorial/visualization_tutorial/minpy_visualization.html).

#### What does it let you do?

As we said, it is an interface for numpy and MXNet. So it is a perfect tool to perform matrix operations and define neural networks models in a simple way as in Keras.
You can choose execute the code on the CPU or the GPU by just changing the context in a simple way.
```python
# Choosing gpu
context.set_context(context.gpu(0)

# Defining a network

class TwoLayerNet(minpy.nn.model.ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (batch_size, flattened_input_size))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Second affine layer.
        y3 = layers.affine(y2, self.params['w2'], self.params['b2'])
        return y3

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_loss(predict, y)
```

By default, MXNet implementations for GPU operations will be run. Any fallback will run transparently on the CPU.
During the development of this proof of concept, the limitations and pitfalls were:

* Not support in-place array operations
* Force to use MinPy consistantly
* Not support all numpy submodules

Anyway, for the limitations I recommend to check their if there were any update. [docs](http://minpy.readthedocs.io/en/latest/feature/limitation.html).

#### How faster is than numpy?

###### Dot product comparison

```python

with cpu():
    x_cpu = random.rand(1024, 1024) - 0.5
    y_cpu = random.rand(1024, 1024) - 0.5

    t0 = time.time()
    for i in xrange(n):
        z_cpu = np.dot(x_cpu, y_cpu)
    z_cpu.asnumpy()
    t1 = time.time()

with gpu(0):
    x_gpu0 = random.rand(1024, 1024) - 0.5
    y_gpu0 = random.rand(1024, 1024) - 0.5

    t2 = time.time()
    for i in xrange(n):
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()
    t3 = time.time()

```

* **CPU** Performance: **0.016882** s/iter
* **GPU** Performance: **0.001476** s/iter

But remember, **GPU is not always faster than CPU**, no matter the framework you use.

```python

from scipy import misc
face = misc.imread('face.png') # Classic numpy array
face_minpy = np.array(face) # minpy array

with cpu():
    t1 = time.time()
    for i in xrange(n):
        gray_cpu = np.dot(face, np.array([0.299, 0.587, 0.114]))
    t2 = time.time()

with gpu(0):
    t3 = time.time()
    for i in xrange(n):
        gray_gpu0 = np.dot(face_minpy, np.array([0.299, 0.587, 0.114]))
    t4 = time.time()

```
* **CPU** Performance: **0.009076** s/iter
* **GPU** Performance: **0.889298** s/iter
