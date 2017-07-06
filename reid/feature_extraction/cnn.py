from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


# Read features extracted by legacy code
# In the feature_dir, each image file name has a corresponding feature file with feature_ext
def read_feature_from_disk(feature_dir, fnames, feature_ext='txt'):
    import os.path as osp
    import numpy as np
    features = []
    for fname in fnames:
        feat_fpath = osp.splitext(osp.basename(fname))[0] + '.' + feature_ext
        feat_fpath = osp.join(feature_dir, feat_fpath)
        if not osp.isfile(feat_fpath):
            raise RuntimeError("File not exists: {} "
                               .format(feat_fpath))
        features.append(np.loadtxt(feat_fpath))

    # Convert to pytoch float tensor which can be consumed by evaluator
    features = to_torch(np.asanyarray(features))

    return features
