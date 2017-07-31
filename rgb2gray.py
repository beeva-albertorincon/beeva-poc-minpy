import minpy.numpy as np
import minpy.numpy.random as random
from minpy.context import cpu, gpu
import time
from scipy import misc

face = misc.imread('face.png')
face_minpy = np.array(face)

n = 5

with cpu():
    t1 = time.time()
    for i in xrange(n):
        gray_cpu = np.dot(face, np.array([0.299, 0.587, 0.114]))
    t2 = time.time()
print("run on cpu: %.6f s/iter" % ((t2 - t1) / n))
#
with gpu(0):
    t3 = time.time()
    for i in xrange(n):
        gray_gpu0 = np.dot(face_minpy, np.array([0.299, 0.587, 0.114]))
    t4 = time.time()
print("run on gpu: %.6f s/iter" % ((t4 - t3) / n))
