import numpy as np
import h5py
import caffe

MODEL_FILE = 'model_deploy.prototxt'
PRETRAINED_FILE = 'model_pretrained.caffemodel'

with h5py.File('data/validate.h5', 'r') as fin:
    inputs = fin['data'][:]

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
out = net.forward(**{net.inputs[0]: inputs})
labels = np.argmax(out[net.outputs[0]].squeeze(), axis=1)

np.savetxt('out/validation_labels.txt', labels, fmt="%i")
