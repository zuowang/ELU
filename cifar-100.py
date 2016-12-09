
# coding: utf-8

# # CIFAR-100 State-of-Art Model
# ----
# 
# In this example, we will show a new state-of-art result on CIFAR-100.
# We use a sub-Inception Network with Randomized ReLU (RReLU), and achieved 75.68% accuracy on CIFAR-100.
# 
# We trained from raw pixel directly, only random crop from 3x28x28 from original 3x32x32 image with random flip, which is same to other experiments. 
# 
# We don't do any parameter search, all hyper-parameters come from ImageNet experience, and this work is just for fun. Definitely you can improve it.
# 
# Train this network requires 3796MB GPU Memory.
# 
# ----
# 
# 
# | Model                       | Test Accuracy |
# | --------------------------- | ------------- |
# | **Sub-Inception + RReLU** [1], [2]   | **75.68%**       |
# | Highway Network  [3] | 67.76%        |
# | Deeply Supervised Network [4]   | 65.43%        |
# 
# 

# In[1]:

import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# Next step we will set up basic Factories for Inception

# In[2]:

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.LeakyReLU(data=bn, act_type='rrelu', name='rrelu_%s%s' %(name, suffix))
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


# Build Network by using Factories

# In[3]:

def inception(nhidden, grad_scale):
    # data
    data = mx.symbol.Variable(name="data")
    # stage 2
    in3a = InceptionFactoryA(data, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    # linear classifier
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=nhidden, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

softmax = inception(100, 1.0)


# Make data iterator. Note we convert original CIFAR-100 dataset into image format then pack into RecordIO in purpose of using our build-in image augmentation. For details about RecordIO, please refer ()[]

# In[4]:

batch_size = 64

train_dataiter = mx.io.ImageRecordIter(
    shuffle=True,
    path_imgrec="./data/train.rec",
    mean_img="./data/mean.bin",
    rand_crop=True,
    rand_mirror=True,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2)

test_dataiter = mx.io.ImageRecordIter(
    path_imgrec="./data/test.rec",
    mean_img="./data/mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)


# Make model

# In[ ]:

num_epoch = 38
model_prefix = "model/cifar_100"

softmax = inception(100, 1.0)

model = mx.model.FeedForward(ctx=[mx.gpu(i) for i in range(8)], symbol=softmax, num_epoch=num_epoch,
                             learning_rate=0.05, momentum=0.9, wd=0.0001)


# Fit first stage

# In[ ]:

model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size, 200),
          epoch_end_callback=mx.callback.do_checkpoint(model_prefix))


# Without reducing learning rate, this model is able to achieve state-of-art result.
# 
# Let's reduce learning rate to train a few more rounds.
# 

# In[ ]:

# load params from saved model
num_epoch = 38
model_prefix = "model/cifar_100"
tmp_model = mx.model.FeedForward.load(model_prefix, epoch)

# create new model with params
num_epoch = 6
model_prefix = "model/cifar_100_stage2"
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
                             learning_rate=0.01, momentum=0.9, wd=0.0001,
                             arg_params=tmp_model.arg_params, aux_params=tmp_model.aux_params,)


model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size, 200),
          epoch_end_callback=mx.callback.do_checkpoint(model_prefix))


# 

# **Reference**
# 
# [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
# 
# [2] Xu, Bing, et al. "Empirical Evaluation of Rectified Activations in Convolutional Network." arXiv preprint arXiv:1505.00853 (2015).
# 
# [3] Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway Networks." arXiv preprint arXiv:1505.00387 (2015).
# 
# [4] Lee, Chen-Yu, et al. "Deeply-supervised nets." arXiv preprint arXiv:1409.5185 (2014).

# In[ ]:



