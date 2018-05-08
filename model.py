import tensorflow as tf


def conv_layer(inputs, filters, kernel_size, name, strides=(1,1), padding='same',
                is_training=False, is_separable=False):
    """
    Create a convolutional layer with the given layer as input.

    :param inputs: <Tensor>
        The Tensor that acts as input into this layer.
    :param filters: <int>
        Number of filters in the convolution layer.
    :param kernel_size <int or tuple>
        A tuple or list of 2 integers specifying the spatial dimensions of the filters.
    :param name <string>
        Name of the layer.
    :param strides <int or tuple>
        List of 2 positive integers specifying the strides of the convolution.
    :param padding <string>
        One of "valid" or "same".
    :param is_training: <bool>
        True if the model is training and weights can be updated.
    :param is_separable: <bool>
        True if the convolution should be a separable convolution.

    :returns Tensor
        A new convolutional layer
    """
    with tf.name_scope(name):
        if(is_separable):
            conv_layer = tf.layers.separable_conv2d(inputs, filters, kernel_size, strides,
                            padding, use_bias=False, activation=None, name=name+'_sepa',
                            depthwise_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            depthwise_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        else:
            conv_layer = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding,
                            use_bias=False, activation=None, name=name+'_conv',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training, name=name+'_conv_batch')
        conv_layer = tf.nn.elu(conv_layer, name=name+'_conv_layer_elu')
        return conv_layer


def inception2d(inputs, filters, is_training, name):
    '''
    Inception V3 layer

    :param inputs: <Tensor>
        The Tensor that acts as input into this layer
    :param filters: <int or (tuple)>
        Number of filters per convolution layer
    :param is_training: <bool>
        True if the model is training and weights can be updated
    :param name <string>
        Name of the layer

    return <Tensor>
        Activated concatination of the inception layer
    '''
    with tf.name_scope(name):
        # 1x1
        conv_1 = conv_layer(inputs, filters, 1, name=name+'_ins_conv_1', is_training=is_training)

        # 1x1 -> 3x3
        conv_2 = conv_layer(inputs, filters, 1, name=name+'_ins_conv_2_1', is_training=is_training)
        conv_2 = conv_layer(conv_2, filters, 3, name=name+'_ins_conv_2_2', is_training=is_training)

        # 1x1 -> 3x3 -> 3x3
        conv_3 = conv_layer(inputs, filters, 1, name=name+'_ins_conv_3_1', is_training=is_training)
        conv_3 = conv_layer(conv_3, filters * 1.5, 3, name=name+'_ins_conv_3_2', is_training=is_training)
        conv_3 = conv_layer(conv_3, filters * 2, 3, name=name+'_ins_conv_3_3', is_training=is_training)

        output = tf.concat([conv_1, conv_2, conv_3], axis=3)  # Concat in the 4th dim to stack
        output = tf.layers.batch_normalization(output, name=name+'_ins_batch', training=is_training)
        return tf.nn.elu(output, name=name+'_ins_final_elu')


def basicRU(inputs, filters, name, is_separable=False, is_training=False):

    with tf.name_scope(name):
        conv_1 = conv_layer(inputs, filters, 3, name=name + '_basicRU_conv_1', is_separable=is_separable,
                            is_training=is_training)

        conv_2 = conv_layer(conv_1, filters, 3, name=name + '_basicRU_conv_2', is_separable=is_separable,
                            is_training=is_training)

        return conv_2



def residualUnit(inputs, filters, name, is_separable=False, is_training=False):
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

        # perform a separable convolution without pooling
        conv_1 = conv_layer(inputs, filters, 1, name=name+'_RU_conv_1_1', is_separable=is_separable, is_training=is_training)

        conv_2 = conv_layer(inputs, filters, 1, name=name+'_RU_conv_2_1', is_separable=is_separable, is_training=is_training)
        conv_2 = conv_layer(conv_2, filters, 3, name=name+'_RU_conv_2_2', is_separable=is_separable, is_training=is_training)
        conv_2 = conv_layer(conv_2, filters, 3, name=name+'_RU_conv_2_3', is_separable=is_separable, is_training=is_training)

        return tf.add(conv_1, conv_2, name=name+'_RU_conv_add')


def encoder(rs, ps, filters, id, loops, name, is_encoder=True, is_training=False):
    '''
    FRRU module which consists of taking the residual stream and the previous
    FRRU blocks, performs the set layout and returns the output for the next block.
    When in the encoder newtwork it will max pool the input and when in the
    decoder network it will un pool the input.

    :param rs: <Tensor>
        The Tensor from the residual stream.
    :param ps: <Tensor>
        The Tensor from the previous FRRU.
    :param filters: <int>
        Number of filters in the convolution layer.
    :param id <string>
        A label id for the network block, used for tensor names.
    :param name <string>
        Name of the layer
    :param is_encoder: <bool>
        True if the block is in the encoder network, max pooling will be performed
        on the previous FRRU input. False for decoder network and un pooling on
        the previous FRRUthe input.
    :param is_training: <bool>
        True if the model is training and weights can be updated.

    :returns Tensor, Tensor
        The top level unchanged image data and the next blocks prev_inputs.
    '''
    with tf.name_scope(name):

        ps_pool = ps
        if is_encoder:
            ps_pool = tf.layers.max_pooling2d(ps, 2, (2,2), padding='same', name=name+'_max_pool_enc_'+id)
        else:
            # in the decoder so we need to unpool
            ps_pool = tf.layers.conv2d_transpose(ps, filters, 4, strides=(2,2),
                                padding='same', name=name+'_trans_conv_enc_pool_'+id,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            ps_pool = tf.layers.batch_normalization(ps_pool, name=name+'_batch_enc_pool_'+id,
                                                    training=is_training)
            ps_pool = tf.nn.elu(ps_pool, name=name+'_encode_elu_1')

        # pool the residual stream to add into the pooling stream
        rs_pool = rs
        for i in range(1, 1+loops):
            rs_pool = tf.layers.max_pooling2d(rs_pool, 2, (2,2), padding='same',
                                              name=name+'_max_pool_encode_'+id+'_'+str(i))

        ps = tf.concat([rs_pool, ps_pool], axis=3)  # Concat in the 4th dim to stack
        ps = tf.layers.batch_normalization(ps, name=name+'_batch_encode_'+id, training=is_training)

        # perform inception in pool stream 1
        #ps = inception2d(ps, filters, is_training, name=name+'_incept_encode_'+id)
        ps = basicRU(ps, filters, name=name+'_incept_encode_'+id, is_separable=False, is_training=is_training)

        # upsample inception to residual stream
        rs_up = conv_layer(ps, filters*4, 1, name=name+'_conv_1_encode_up_'+id, is_training=is_training)
        loop_ratio = 4 / loops
        for i in range(1, 1+loops):
            filt = int(max((4-(i*loop_ratio))*filters, 32))
            rs_up = tf.layers.conv2d_transpose(rs_up, filt, 4, strides=(2,2),
                                padding='same', name=name+'_trans_conv_encode_up_'+id+'_'+str(i),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            rs_up = tf.layers.batch_normalization(rs_up, name=name+'_batch_norm_encode_up_'+id+'_'+str(i),
                                                  training=is_training)
            rs_up = tf.nn.elu(rs_up, name=name+'_encode_elu_2')

        # combine upsample into residual stream
        rs = tf.add(rs, rs_up, name=name+'_encoder_final_add')

        return rs, ps


def networkModel(inputs, num_classes, name, is_training=False):
    '''
    Model used for semantic segmentation.

    :param inputs: <Tensor>
        Tensor containing the input images.
    :param num_classes: <int>
        The classification classes for the network to predict.
    :param name <string>
        Name of the network model.
    :param is_training: <bool>
        True if the model is training and weights can be updated.

    :returns <Tensor>
        The prediction tensor from the inputs.
    '''

    with tf.name_scope(name):
        rs = residualUnit(inputs, 32, name+'_Network_1_1', is_separable=True, is_training=is_training)
        rs = residualUnit(rs, 32, name+'_Network_1_2', is_separable=True, is_training=is_training)

        # pooling stream
        # Encoder

        # pool stream 1 -> 512x720x32 -> rs = 512x720x32, ps = 256x360x64
        filters = 32
        id = '1'
        loops = 1
        rs, ps = encoder(rs, rs, filters, id, loops, name+'_encode1', is_training=is_training)

        # pool stream 2 -> 256x360x64 -> rs = 512x720x32, ps = 128x180x128
        # downsample
        filters = 64
        id = '2'
        loops = 2
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_encode2', is_training=is_training)

        filters = 128
        id = '3'
        loops = 3
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_encode3', is_training=is_training)

        filters = 256
        id = '4'
        loops = 4
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_encode4', is_training=is_training)

        # decoder
        filters = 128
        id = '5'
        loops = 3
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_decode1', is_encoder=False, is_training=is_training)

        filters = 64
        id = '6'
        loops = 2
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_decode2', is_encoder=False, is_training=is_training)

        filters = 32
        id = '7'
        loops = 1
        rs, ps = encoder(rs, ps, filters, id, loops, name+'_decode3', is_encoder=False, is_training=is_training)

        # unpool the pooling stream to add back into the residual stream
        ps = tf.layers.conv2d_transpose(ps, 48, 4, strides=(2,2),
                            padding='same', name=name+'_trans_conv_ps_pool_last',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        ps = tf.layers.batch_normalization(ps, name=name+'_batch_ps_pool_last_1',
                                                training=is_training)
        ps = tf.nn.elu(ps, name=name+'_ps_final_elu')

        ru = tf.concat([rs, ps], axis=3, name=name+'_RS_PS_concat') #add(rs, ps)

        ru_2 = residualUnit(ru, 32, name+'_RN_2_1', is_separable=False, is_training=is_training)
        ru_2 = residualUnit(ru_2, 16, name+'_RN_2_2', is_separable=False, is_training=is_training)

        # final convolution with a 1x1 and bias with no activation
        output = tf.layers.conv2d(ru_2, num_classes, (1,1), (1,1), 'same',
                        use_bias=True, activation=None, name=name+'_output_conv',
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        return output
