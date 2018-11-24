"""
2018-11-24
"""

from collections import namedtuple
import copy

import tensorflow as tf

slim = tf.contrib.slim

def _make_divisible(v, divisor, min_value=None):
    """make `v` is divided exactly by `divisor`, but keep the min_value"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@slim.add_arg_scope
def _depth_multiplier_func(params,
                           multiplier,
                           divisible_by=8,
                           min_depth=8):
    """get the new channles"""
    if 'num_outputs' not in params:
        return
    d = params['num_outputs']
    params['num_outputs'] = _make_divisible(d * multiplier, divisible_by,
                                                   min_depth)

def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
      Pads the input such that if it was used in a convolution with 'VALID' padding,
      the output would have the same dimensions as if the unpadded input was used
      in a convolution with 'SAME' padding.
      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        rate: An integer, rate for atrous convolution.
      Returns:
        output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                               kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                      [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


@slim.add_arg_scope
def expanded_conv(x,
                  num_outputs,
                  expansion=6,
                  stride=1,
                  rate=1,
                  normalizer_fn=slim.batch_norm,
                  project_activation_fn=tf.identity,
                  padding="SAME",
                  scope=None):
    """The expand conv op in MobileNetv2
        1x1 conv -> depthwise 3x3 conv -> 1x1 linear conv
    """
    with tf.variable_scope(scope, default_name="expanded_conv") as s, \
       tf.name_scope(s.original_name_scope):
        prev_depth = x.get_shape().as_list()[3]
        # the filters of expanded conv
        inner_size = prev_depth * expansion
        net = x
        # only inner_size > prev_depth, use expanded conv
        if inner_size > prev_depth:
            net = slim.conv2d(net, inner_size, 1, normalizer_fn=normalizer_fn,
                              scope="expand")
        # depthwise conv
        net = slim.separable_conv2d(net, num_outputs=None, kernel_size=3,
                                    depth_multiplier=1, stride=stride,
                                    rate=rate, normalizer_fn=normalizer_fn,
                                    padding=padding, scope="depthwise")
        # projection
        net = slim.conv2d(net, num_outputs, 1, normalizer_fn=normalizer_fn,
                          activation_fn=project_activation_fn, scope="project")

        # residual connection
        if stride == 1 and net.get_shape().as_list()[-1] == prev_depth:
            net += x

        return net

def global_pool(x, pool_op=tf.nn.avg_pool):
    """Applies avg pool to produce 1x1 output.
    NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
        baked in average pool which has better support across hardware.
    Args:
        input_tensor: input tensor
        pool_op: pooling op (avg pool is default)
    Returns:
        a tensor batch_size x 1 x 1 x depth.
    """
    shape = x.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size = tf.convert_to_tensor(
            [1, tf.shape(x)[1], tf.shape(x)[2], 1])
    else:
        kernel_size = [1, shape[1], shape[2], 1]
    output = pool_op(x, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output


_Op = namedtuple("Op", ['op', 'params', 'multiplier_func'])

def op(op_func, **params):
    return _Op(op=op_func, params=params,
               multiplier_func=_depth_multiplier_func)


CONV_DEF = [op(slim.conv2d, num_outputs=32, stride=2, kernel_size=3),
            op(expanded_conv, num_outputs=16, expansion=1),
            op(expanded_conv, num_outputs=24, stride=2),
            op(expanded_conv, num_outputs=24, stride=1),
            op(expanded_conv, num_outputs=32, stride=2),
            op(expanded_conv, num_outputs=32, stride=1),
            op(expanded_conv, num_outputs=32, stride=1),
            op(expanded_conv, num_outputs=64, stride=2),
            op(expanded_conv, num_outputs=64, stride=1),
            op(expanded_conv, num_outputs=64, stride=1),
            op(expanded_conv, num_outputs=64, stride=1),
            op(expanded_conv, num_outputs=96, stride=1),
            op(expanded_conv, num_outputs=96, stride=1),
            op(expanded_conv, num_outputs=96, stride=1),
            op(expanded_conv, num_outputs=160, stride=2),
            op(expanded_conv, num_outputs=160, stride=1),
            op(expanded_conv, num_outputs=160, stride=1),
            op(expanded_conv, num_outputs=320, stride=1),
            op(slim.conv2d, num_outputs=1280, stride=1, kernel_size=1),
            ]


def mobilenet_arg_scope(is_training=True,
                        weight_decay=0.00004,
                        stddev=0.09,
                        dropout_keep_prob=0.8,
                        bn_decay=0.997):
    """Defines Mobilenet default arg scope.
    Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)
     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
    Args:
        is_training: if set to False this will ensure that all customizations are
            set to non-training mode. This might be helpful for code that is reused
        across both training/evaluation, but most of the time training_scope with
        value False is not needed. If this is set to None, the parameters is not
        added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob: dropout keep probability (not set if equals to None).
        bn_decay: decay for the batch norm moving averages (not set if equals to
            None).
    Returns:
        An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': bn_decay,
        'is_training': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu6), \
        slim.arg_scope([slim.batch_norm], **batch_norm_params), \
        slim.arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
        slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                       biases_initializer=None,
                       padding="SAME"), \
        slim.arg_scope([slim.conv2d],
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
        slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
        return s


def mobilenetv2(x,
                num_classes=1001,
                depth_multiplier=1.0,
                scope='MobilenetV2',
                finegrain_classification_mode=False,
                min_depth=8,
                divisible_by=8,
                output_stride=None,
                ):
    """Mobilenet v2
    Args:
        x: The input tensor
        num_classes: number of classes
        depth_multiplier: The multiplier applied to scale number of
            channels in each layer. Note: this is called depth multiplier in the
            paper but the name is kept for consistency with slim's model builder.
        scope: Scope of the operator
        finegrain_classification_mode: When set to True, the model
            will keep the last layer large even for small multipliers.
            The paper suggests that it improves performance for ImageNet-type of problems.
        min_depth: If provided, will ensure that all layers will have that
          many channels after application of depth multiplier.
       divisible_by: If provided will ensure that all layers # channels
          will be divisible by this number.
    """
    conv_defs = CONV_DEF

    # keep the last conv layer very larger channel
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs[-1].params['num_outputs'] /= depth_multiplier

    depth_args = {}
    # NB: do not set depth_args unless they are provided to avoid overriding
    # whatever default depth_multiplier might have thanks to arg_scope.
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

    with slim.arg_scope([_depth_multiplier_func], **depth_args):
        with tf.variable_scope(scope, default_name='Mobilenet'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            net = x
            # Insert default parameters before the base scope which includes
            # any custom overrides set in mobilenet.
            end_points = {}
            scopes = {}
            for i, opdef in enumerate(conv_defs):
                params = dict(opdef.params)
                opdef.multiplier_func(params, depth_multiplier)
                stride = params.get('stride', 1)
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= stride
                else:
                    layer_stride = stride
                    layer_rate = 1
                    current_stride *= stride
                # Update params.
                params['stride'] = layer_stride
                # Only insert rate to params if rate > 1.
                if layer_rate > 1:
                    params['rate'] = layer_rate

                try:
                    net = opdef.op(net, **params)
                except Exception:
                    raise ValueError('Failed to create op %i: %r params: %r' % (i, opdef, params))

            with tf.variable_scope('Logits'):
                net = global_pool(net)
                end_points['global_pool'] = net
                if not num_classes:
                    return net, end_points
                net = slim.dropout(net, scope='Dropout')
                # 1 x 1 x num_classes
                # Note: legacy scope name.
                logits = slim.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    biases_initializer=tf.zeros_initializer(),
                    scope='Conv2d_1c_1x1')

                logits = tf.squeeze(logits, [1, 2])

                return logits


if __name__ == "__main__":
    import cv2
    import numpy as np

    inputs = tf.placeholder(tf.uint8, [None, None, 3])
    images = tf.expand_dims(inputs, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (224, 224))

    with slim.arg_scope(mobilenet_arg_scope(is_training=False)):
        logits = mobilenetv2(images)

    # Restore using exponential moving average since it produces (1.5-2%) higher
    # accuracy
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()

    saver = tf.train.Saver(vars)

    print(len(tf.global_variables()))
    for var in tf.global_variables():
        print(var)
    checkpoint_path = r"C:\Users\xiaoh\Desktop\temp\mobilenet_v2_1.0_224\mobilenet_v2_1.0_224.ckpt"
    image_file = "C:/Users/xiaoh/Desktop/temp/pandas.jpg"
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(np.argmax(sess.run(logits, feed_dict={inputs: img})[0]))










