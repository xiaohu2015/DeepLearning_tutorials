# Deep Learning Tutorials with Tensorflow
The deeplearning algorithms are carefully implemented by [tensorflow](https://www.tensorflow.org/).  
### Environment
- Python 3.5
- tensorflow 1.4
- pytorch 0.2.0

### The deeplearning algorithms includes (now):
- Logistic Regression  [logisticRegression.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/logisticRegression.py)
- Multi-Layer Perceptron (MLP) [mlp.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/mlp.py)
- Convolution Neural Network (CNN) [cnn.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/cnn.py)
- Denoising Aotoencoder (DA) [da.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/da.py)
- Stacked Denoising Autoencoder (SDA) [sda.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/sda.py)
- Restricted Boltzmann Machine (RBM) [[rbm.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/rbm.py)    [gbrbm.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/gbrbm.py)]
- Deep Belief Network (DBN) [dbn.py](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/dbn.py)

Note: the project aims at imitating the well-implemented algorithms in [Deep Learning Tutorials](http://www.deeplearning.net/tutorial/) (coded by [Theano](http://deeplearning.net/software/theano/index.html)).

### CNN Models
- MobileNet [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py) [paper](https://arxiv.org/abs/1704.04861) [ref](https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py)]
- SqueezeNet [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/SqueezeNet.py) [paper](https://arxiv.org/abs/1602.07360)]
- ResNet [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/ResNet50.py) [caffe ref](https://github.com/KaimingHe/deep-residual-networks) [paper1](https://arxiv.org/abs/1512.03385) [paper2](https://arxiv.org/abs/1603.05027)]
- ShuffleNet [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/ShuffleNet.py) by pytorch [paper](http://cn.arxiv.org/pdf/1707.01083v2)]
- DenseNet [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/densenet.py) [pytorch_ref](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) [paper](https://arxiv.org/abs/1608.06993)]

### Object detection
- YOLOv1 [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/ObjectDetections/yolo/yolo_tf.py) [paper](https://arxiv.org/abs/1506.02640) [ref](https://github.com/gliese581gg/YOLO_tensorflow)]
- SSD [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/ObjectDetections/SSD/SSD_demo.py) [paper](https://arxiv.org/pdf/1611.10012.pdf) [slides](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf) [cafe](https://github.com/weiliu89/caffe/tree/ssd) [TF](https://arxiv.org/abs/1512.02325) [pytorch](https://github.com/amdegroot/ssd.pytorch) ]
- YOLOv2 [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/ObjectDetections/yolo2) [paper](https://arxiv.org/abs/1612.08242) [ref](https://github.com/yhcc/yolo2)]

### Practical examples
You can find more practical examples with tensorflow here:
- CNN for setence classification [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/cnn_setence_classification)] [[blog](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)] [[paper](https://arxiv.org/pdf/1408.5882v2.pdf)]
- RNN for language model [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/rnn_language_model)] [[blog](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)] [[blog_cn](http://blog.csdn.net/xiaohu2022/article/details/54578013)]
- LSTM for language model (PTB data) [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/lstm_model_ptb)] [[tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/recurrent/index.html#recurrent-neural-networks)] [[paper](https://arxiv.org/pdf/1409.2329.pdf)]
- VGG model for image classification (object recongnition) [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/VGG)] [[source](https://github.com/machrisaa/tensorflow-vgg)]
- Residual network for cifar10_dataset [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/Resnet)] [[source](https://github.com/wenxinxu/resnet-in-tensorflow)] [[paper](https://arxiv.org/pdf/1603.05027v3.pdf)]
- LSTM for time series prediction [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/examples/lstm_time_series_regression)] [[source](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2.2/full_code.py)]
- Generative adversarial network (GAN) [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/examples/gan)]
- Variational autoencoder (VAE) [[self](https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/examples/VAE)]

### Results
![1](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/filters_corruption_30.png)
![2](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/new_filters_at_epoch_14.png)
![3](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/new_original_and_10samples.png)
![4](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/DBN_results.png)
![5](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/examples/lstm_time_series_regression/lstm_regression_results.png)

### Fun Blogs
- [Chatbots with Seq2Seq](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

### Personal Notes
- Tensorflow for RNNs [[tf_rnn.ipynb](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/notes/tf_rnn.ipynb)]
- Tensorflow for Autoencoder [[tf_autoencoder.ipynb](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/notes/tf_autoencoder.ipynb)]

### Other Tutorials
- [ageron/handson-ml
](https://github.com/ageron/handson-ml/)
- [Hvass-Labs/TensorFlow-Tutorials
](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [BinRoot/TensorFlow-Book
](https://github.com/BinRoot/TensorFlow-Book)
- [sjchoi86/dl_tutorials_10weeks
](https://github.com/sjchoi86/dl_tutorials_10weeks)

#### Don't hesitate to star this project if it is helpful!
### If you benefit from the tutorial, please make a small donation by WeChat sweep.
![weichat](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/weichat.jpg)
## 微信号：xiaoxiaohu1994
## 欢迎关注微信公众号：机器学习算法全栈工程师(Jeemy110)
![公众号](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/results/654362565405877642.jpg)
