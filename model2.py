import tensorflow as tf

def variable_summaries(var, name):
    '''
    Creates summaries of the passed in variable for TensorBoard visualization
    from TensorBoard: Visualizing Learning
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    :param var: Variable that will have its values saved for visualization
    :param name: A label name for the variable
    :return:
    '''

    with tf.name_scope('summaries_' +name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_' +name, mean) # save as a tensorboard summary
        with tf.name_scope('stddev_' +name):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_' +name, stddev)
        tf.summary.scalar('max_' +name, tf.reduce_max(var))
        tf.summary.scalar('min_' +name, tf.reduce_min(var))
        tf.summary.histogram('histogram_' +name, var)


def batch_normalisation(layer, filter_size, is_training, name='batch_norm'):

    gamma = tf.Variable(tf.ones([filter_size]))
    variable_summaries(gamma, 'gamma_' + name)
    beta = tf.Variable(tf.zeros([filter_size]))
    variable_summaries(beta, 'beta_' + name)

    pop_mean = tf.Variable(tf.zeros([filter_size]), trainable=False)
    variable_summaries(pop_mean, 'pop_mean_' + name)
    pop_variance = tf.Variable(tf.ones([filter_size]), trainable=False)
    variable_summaries(pop_variance, 'pop_variance_' + name)

    epsilon = 1e-3

    def batch_norm_training():
        # Important to use the correct dimensions here to ensure the mean and variance are calculated
        # per feature map instead of for the entire layer
        batch_mean, batch_variance = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon, name=name)

    def batch_norm_inference():
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon, name=name)

    batch_norm = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    tf.summary.histogram('batch_norm_' + name, batch_norm)

    return batch_norm


def pooling(x, name='max_pool'):
    # return the max pool of the activation
    ksize = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    return tf.nn.max_pool(x, ksize, pool_strides, padding='SAME', name=name)


def convolution(x, filter_size, is_training, ksize=(3,3), strides=(1,1), padding='SAME', name='conv', use_bias=True, alpha=0.01):
    '''
    Convolution layer where each layer will max pool with kernel size (2,2) and strides (2,2)
    this will result in halving the convolutioal layer
    :param x: Input features from the preceding layer
    :param filter_size: Number of filter output from the convolution
    :param ksize: Kernal size 2D Tuple with kernel width and height
    :param strides: Kernel stride movement. 2D Tuple of width and height
    :param padding: Convolution padding type
    :param name: A label name for the layer
    :return: A convolution tensor of the input data
    '''

    with tf.name_scope(name):
        mean = 0
        sigma = 0.1

        #input_features = x.get_shape().as_list()[-1] # number of features
        shape = x.get_shape().as_list()

        # define the kernel shape for the convolutional filter
        shape = [ksize[0], ksize[1], shape[-1], filter_size]
        weight = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=sigma), name='conv_weight')
        variable_summaries(weight, 'weight_' + name) # save the weight summary for tensorboard

        # perform a convolution layer on the x_data using strides conv_strides
        stride = [1, strides[0], strides[1], 1]
        conv = tf.nn.conv2d(x, weight, strides=stride, padding=padding, name=name+'_conv')

        if use_bias:
            bias = tf.Variable(tf.constant(0.1, shape=[filter_size]), name='conv_bias')
            variable_summaries(bias, 'bias_' + name)  # save the weight summary for tensorboard

            conv = tf.nn.bias_add(conv, bias)
        else:
            conv = batch_normalisation(conv, filter_size, is_training, name=name+'_bn')

        # Leaky ReLU activation
        lrelu = tf.maximum(alpha * conv, conv)
        tf.summary.histogram('activation_' + name, lrelu)

        return lrelu


def deconvolution(x, filter_size, is_training, padding='same', name='deconv', alpha=0.01):
    '''
    Deconvolution layer where each layer will max pool with kernel size (2,2) and strides (2,2)
    this will result in halving the convolutioal layer
    :param x: Input features from the preceding layer
    :param filter_size: Number of filter output from the convolution
    :param ksize: Kernal size 2D Tuple with kernel width and height
    :param strides: Kernel stride movement. 2D Tuple of width and height
    :param padding: Convolution padding type
    :param name: A label name for the layer
    :return: A convolution tensor of the input data
    '''

    with tf.name_scope(name):
        deconv = tf.layers.conv2d_transpose(x, filter_size, 2, strides=2,
                                             padding=padding, name=name+'_dconv',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        batch_norm = batch_normalisation(deconv, filter_size, is_training, name=name+'_bn')

        # Leaky ReLU activation
        lrelu = tf.maximum(alpha * batch_norm, batch_norm)
        tf.summary.histogram('activation_' + name, lrelu)

        return lrelu


def residualUnit(inputs, filters, name, is_training=False):
    '''
    Implements a residual network.

    :param inputs: <Tensor>
        The Tensor that acts as input into this layer.
    :param filters: <int>
        Number of filters in the convolution layer.
    :param name <string>
        Name of the layer.
    :param is_separable: <bool>
        True if the convolution should be a separable convolution or a normal
        convolution.
    :param is_training: <bool>
        True if the model is training and weights can be updated.

    return <Tensor>
        The downsampled convolution
    '''
    with tf.name_scope(name):
        conv_1 = convolution(inputs, filters, is_training, ksize=(1,1), use_bias=False, name=name+'_RU_conv_1_1')

        conv_2 = convolution(inputs, filters, is_training, ksize=(1, 1), use_bias=False, name=name + '_RU_conv_2_1')
        conv_2 = convolution(conv_2, filters, is_training, ksize=(3, 3), use_bias=False, name=name + '_RU_conv_2_2')
        conv_2 = convolution(conv_2, filters, is_training, ksize=(3, 3), use_bias=False, name=name + '_RU_conv_2_3')

        return tf.add(conv_1, conv_2, name=name+'_RU_conv_add')


def encoder(ps, rs, filters, loops, name, is_training=False, is_encoder=True):

    with tf.name_scope(name):
        ps_pool = ps
        if is_encoder:
            ps_pool = pooling(ps_pool)
        else:
            # in the decoder so we need to unpool
            ps_pool = deconvolution(ps_pool, filters, is_training=is_training, name=name + 'ps_deconv_1')

        # pool the residual stream to add into the pooling stream
        rs_pool = rs
        for i in range(1, 1 + loops):
            rs_pool = pooling(rs_pool)

        ps_pool = tf.concat([rs_pool, ps_pool], axis=3)  # Concat in the 4th dim to stack
        #ps_pool = batch_normalisation(ps_pool, filters, is_training=is_training, name=name + '_bn')

        ps_pool = convolution(ps_pool, filters, is_training=is_training, ksize=(3, 3), use_bias=False, name=name + '_EN_conv_1_1')
        ps_pool = convolution(ps_pool, filters, is_training=is_training, ksize=(3, 3), use_bias=False, name=name + '_EN_conv_1_2')

        # upsample inception to residual stream
        rs_up = convolution(ps_pool, filters, is_training=is_training, ksize=(1, 1), use_bias=True, name=name + '_EN_up_conv_2')
        loop_ratio = 4 / loops
        for i in range(1, 1 + loops):
            filt = int(max((4 - (i * loop_ratio)) * filters, 32))
            rs_up = deconvolution(rs_up, filt, is_training=is_training, name=name + 'deconv'+ '_' + str(i))

        # combine upsample into residual stream
        rs = tf.add(rs, rs_up, name=name + '_encoder_final_add')

        print("rs_pool shape : ", rs.get_shape())
        print("ps_pool shape : ", ps_pool.get_shape())

        return ps_pool, rs


def model(inputs, num_classes, is_training, name='model'):
    '''
    Create the neural model for learning the data. The model used is a two layer
    convolution with a two layer fully connected output
    :param x: Input features from the preceding layer
    :param name: A label name for the model
    :return: Tensor of n_classes containing the model predictions
    '''

    with tf.name_scope(name):
        # Input: 28x28x3, Output: 14x14x6
        x = residualUnit(inputs, 32, name='RS_1_1', is_training=is_training)
        x = residualUnit(x, 32, name='RS_1_2', is_training=is_training)

        # Input: 14x14x6, Output: 7x7x16
        ps, rs = encoder(x, x, filters=64, loops=1, name='encode_1', is_training=is_training, is_encoder=True)
        #x = residualUnit(x, 64, name='RS_2', is_training=is_training)
        #x = pooling(x)

        ps, rs = encoder(ps, rs, filters=128, loops=2, name='encode_2', is_training=is_training, is_encoder=True)
        #x = residualUnit(x, 128, name='RS_3', is_training=is_training)
        #skip_2 = pooling(x)

        ps, rs = encoder(ps, rs, filters=256, loops=3, name='encode_3', is_training=is_training, is_encoder=True)
        #x = residualUnit(skip_2, 256, name='RS_4', is_training=is_training)
        #x = pooling(x)

        ps, rs = encoder(ps, rs, filters=384, loops=4, name='encode_4', is_training=is_training, is_encoder=True)
        #x = residualUnit(x, 384, name='RS_5', is_training=is_training)
        #x = pooling(x)

        #fcu = convolution(x, 384, is_training, ksize=(1, 1), name='FCU')

        ps, rs = encoder(ps, rs, filters=256, loops=3, name='decode_1', is_training=is_training, is_encoder=False)
        #x = deconvolution(ps, 256, is_training, name='deconv1')

        ps, rs = encoder(ps, rs, filters=128, loops=2, name='decode_2', is_training=is_training, is_encoder=False)
        #x = deconvolution(x, 128, is_training, name='deconv2')
        #x = tf.add(x, skip_2, name="skip_2_add")

        ps, rs = encoder(ps, rs, filters=64, loops=1, name='decode_3', is_training=is_training, is_encoder=False)
        #x = deconvolution(x, 64, is_training, name='deconv3')

        #ps, rs = encoder(ps, rs, filters=32, loops=1, name='decode_1', is_training=is_training, is_encoder=False)
        ps = deconvolution(ps, 32, is_training, name='deconv_4')
        x = tf.add(ps, rs, name="skip_1_add")

        output = residualUnit(x, num_classes, name='RS_out', is_training=is_training)

        return output