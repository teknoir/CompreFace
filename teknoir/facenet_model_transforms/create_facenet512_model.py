import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
from deepface import DeepFace
import numpy as np

def batch_norm(inputs, phase_train, scope):
    """Batch normalization with TF1.x compatibility"""
    return tf1.layers.batch_normalization(
        inputs=inputs,
        training=phase_train,
        name=scope,
        fused=True
    )

def conv2d(inputs, filters, kernel_size, stride=1, scope=None, padding='SAME'):
    """Convolution with TF1.x compatibility"""
    with tf1.variable_scope(scope):
        net = tf1.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name='Conv2d'
        )
        return net

def block35(net, scale=1.0, phase_train=True, scope=None):
    """35x35 resnet block"""
    with tf1.variable_scope(scope):
        tower_conv = conv2d(net, 32, 1, scope='Branch_0/Conv2d_1x1')
        tower_conv = batch_norm(tower_conv, phase_train, 'Branch_0/Conv2d_1x1/BatchNorm')
        tower_conv = tf1.nn.relu(tower_conv)
        
        tower_conv1_0 = conv2d(net, 32, 1, scope='Branch_1/Conv2d_0a_1x1')
        tower_conv1_0 = batch_norm(tower_conv1_0, phase_train, 'Branch_1/Conv2d_0a_1x1/BatchNorm')
        tower_conv1_0 = tf1.nn.relu(tower_conv1_0)
        tower_conv1_1 = conv2d(tower_conv1_0, 32, 3, scope='Branch_1/Conv2d_0b_3x3')
        tower_conv1_1 = batch_norm(tower_conv1_1, phase_train, 'Branch_1/Conv2d_0b_3x3/BatchNorm')
        tower_conv1_1 = tf1.nn.relu(tower_conv1_1)
        
        tower_conv2_0 = conv2d(net, 32, 1, scope='Branch_2/Conv2d_0a_1x1')
        tower_conv2_0 = batch_norm(tower_conv2_0, phase_train, 'Branch_2/Conv2d_0a_1x1/BatchNorm')
        tower_conv2_0 = tf1.nn.relu(tower_conv2_0)
        tower_conv2_1 = conv2d(tower_conv2_0, 32, 3, scope='Branch_2/Conv2d_0b_3x3')
        tower_conv2_1 = batch_norm(tower_conv2_1, phase_train, 'Branch_2/Conv2d_0b_3x3/BatchNorm')
        tower_conv2_1 = tf1.nn.relu(tower_conv2_1)
        tower_conv2_2 = conv2d(tower_conv2_1, 32, 3, scope='Branch_2/Conv2d_0c_3x3')
        tower_conv2_2 = batch_norm(tower_conv2_2, phase_train, 'Branch_2/Conv2d_0c_3x3/BatchNorm')
        tower_conv2_2 = tf1.nn.relu(tower_conv2_2)
        
        mixed = tf1.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = conv2d(mixed, net.get_shape()[3], 1, scope='Conv2d_1x1')
        up = batch_norm(up, phase_train, 'Conv2d_1x1/BatchNorm')
        net += scale * up
    return net

def block17(net, scale=1.0, phase_train=True, scope=None):
    """17x17 resnet block"""
    with tf1.variable_scope(scope):
        tower_conv = conv2d(net, 128, 1, scope='Branch_0/Conv2d_1x1')
        tower_conv = batch_norm(tower_conv, phase_train, 'Branch_0/Conv2d_1x1/BatchNorm')
        tower_conv = tf1.nn.relu(tower_conv)
        
        tower_conv1_0 = conv2d(net, 128, 1, scope='Branch_1/Conv2d_0a_1x1')
        tower_conv1_0 = batch_norm(tower_conv1_0, phase_train, 'Branch_1/Conv2d_0a_1x1/BatchNorm')
        tower_conv1_0 = tf1.nn.relu(tower_conv1_0)
        tower_conv1_1 = conv2d(tower_conv1_0, 128, [1, 7], scope='Branch_1/Conv2d_0b_1x7')
        tower_conv1_1 = batch_norm(tower_conv1_1, phase_train, 'Branch_1/Conv2d_0b_1x7/BatchNorm')
        tower_conv1_1 = tf1.nn.relu(tower_conv1_1)
        tower_conv1_2 = conv2d(tower_conv1_1, 128, [7, 1], scope='Branch_1/Conv2d_0c_7x1')
        tower_conv1_2 = batch_norm(tower_conv1_2, phase_train, 'Branch_1/Conv2d_0c_7x1/BatchNorm')
        tower_conv1_2 = tf1.nn.relu(tower_conv1_2)
        
        mixed = tf1.concat([tower_conv, tower_conv1_2], 3)
        up = conv2d(mixed, net.get_shape()[3], 1, scope='Conv2d_1x1')
        up = batch_norm(up, phase_train, 'Conv2d_1x1/BatchNorm')
        net += scale * up
    return net

def block8(net, scale=1.0, activation_fn=True, phase_train=True, scope=None):
    """8x8 resnet block"""
    with tf1.variable_scope(scope):
        tower_conv = conv2d(net, 192, 1, scope='Branch_0/Conv2d_1x1')
        tower_conv = batch_norm(tower_conv, phase_train, 'Branch_0/Conv2d_1x1/BatchNorm')
        tower_conv = tf1.nn.relu(tower_conv)
        
        tower_conv1_0 = conv2d(net, 192, 1, scope='Branch_1/Conv2d_0a_1x1')
        tower_conv1_0 = batch_norm(tower_conv1_0, phase_train, 'Branch_1/Conv2d_0a_1x1/BatchNorm')
        tower_conv1_0 = tf1.nn.relu(tower_conv1_0)
        tower_conv1_1 = conv2d(tower_conv1_0, 192, [1, 3], scope='Branch_1/Conv2d_0b_1x3')
        tower_conv1_1 = batch_norm(tower_conv1_1, phase_train, 'Branch_1/Conv2d_0b_1x3/BatchNorm')
        tower_conv1_1 = tf1.nn.relu(tower_conv1_1)
        tower_conv1_2 = conv2d(tower_conv1_1, 192, [3, 1], scope='Branch_1/Conv2d_0c_3x1')
        tower_conv1_2 = batch_norm(tower_conv1_2, phase_train, 'Branch_1/Conv2d_0c_3x1/BatchNorm')
        tower_conv1_2 = tf1.nn.relu(tower_conv1_2)
        
        mixed = tf1.concat([tower_conv, tower_conv1_2], 3)
        up = conv2d(mixed, net.get_shape()[3], 1, scope='Conv2d_1x1')
        up = batch_norm(up, phase_train, 'Conv2d_1x1/BatchNorm')
        if activation_fn:
            net += scale * up
            net = tf1.nn.relu(net)
        else:
            net += up
    return net

def create_inception_resnet_v1(inputs, phase_train=True, embedding_size=512):
    """Creates InceptionResNetV1 architecture with TF1.x compatibility"""
    
    with tf1.variable_scope('InceptionResnetV1'):
        # Initial layers
        net = conv2d(inputs, 32, 3, stride=2, scope='Conv2d_1a_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_1a_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        net = conv2d(net, 32, 3, scope='Conv2d_2a_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_2a_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        net = conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        net = batch_norm(net, phase_train, 'Conv2d_2b_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        # Use strided conv instead of maxpool
        net = conv2d(net, 80, 3, stride=2, scope='Conv2d_3b_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_3b_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        net = conv2d(net, 192, 3, scope='Conv2d_4a_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_4a_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        # Use strided conv instead of maxpool
        net = conv2d(net, 256, 3, stride=2, scope='Conv2d_4b_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_4b_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        # 5x block35 (Inception-ResNet-A block)
        with tf1.variable_scope('Repeat_1'):
            for i in range(5):
                net = block35(net, 0.17, phase_train, f'Block35_{i+1}')
        
        # Reduction-A block
        with tf1.variable_scope('Mixed_6a'):
            tower_conv = conv2d(net, 384, 3, stride=2, scope='Branch_0/Conv2d_1a_3x3', padding='VALID')
            tower_conv = batch_norm(tower_conv, phase_train, 'Branch_0/Conv2d_1a_3x3/BatchNorm')
            tower_conv = tf1.nn.relu(tower_conv)
            
            tower_conv1_0 = conv2d(net, 192, 1, scope='Branch_1/Conv2d_0a_1x1')
            tower_conv1_0 = batch_norm(tower_conv1_0, phase_train, 'Branch_1/Conv2d_0a_1x1/BatchNorm')
            tower_conv1_0 = tf1.nn.relu(tower_conv1_0)
            tower_conv1_1 = conv2d(tower_conv1_0, 192, 3, scope='Branch_1/Conv2d_0b_3x3')
            tower_conv1_1 = batch_norm(tower_conv1_1, phase_train, 'Branch_1/Conv2d_0b_3x3/BatchNorm')
            tower_conv1_1 = tf1.nn.relu(tower_conv1_1)
            tower_conv1_2 = conv2d(tower_conv1_1, 256, 3, stride=2, scope='Branch_1/Conv2d_1a_3x3', padding='VALID')
            tower_conv1_2 = batch_norm(tower_conv1_2, phase_train, 'Branch_1/Conv2d_1a_3x3/BatchNorm')
            tower_conv1_2 = tf1.nn.relu(tower_conv1_2)
            
            net = tf1.concat([tower_conv, tower_conv1_2], 3)
        
        # 10x block17 (Inception-ResNet-B block)
        with tf1.variable_scope('Repeat_2'):
            for i in range(10):
                net = block17(net, 0.10, phase_train, f'Block17_{i+1}')
        
        # Reduction-B block
        with tf1.variable_scope('Mixed_7a'):
            tower_conv = conv2d(net, 256, 1, scope='Branch_0/Conv2d_0a_1x1')
            tower_conv = batch_norm(tower_conv, phase_train, 'Branch_0/Conv2d_0a_1x1/BatchNorm')
            tower_conv = tf1.nn.relu(tower_conv)
            tower_conv_1 = conv2d(tower_conv, 384, 3, stride=2, scope='Branch_0/Conv2d_1a_3x3', padding='VALID')
            tower_conv_1 = batch_norm(tower_conv_1, phase_train, 'Branch_0/Conv2d_1a_3x3/BatchNorm')
            tower_conv_1 = tf1.nn.relu(tower_conv_1)
            
            tower_conv1 = conv2d(net, 256, 1, scope='Branch_1/Conv2d_0a_1x1')
            tower_conv1 = batch_norm(tower_conv1, phase_train, 'Branch_1/Conv2d_0a_1x1/BatchNorm')
            tower_conv1 = tf1.nn.relu(tower_conv1)
            tower_conv1_1 = conv2d(tower_conv1, 256, 3, stride=2, scope='Branch_1/Conv2d_1a_3x3', padding='VALID')
            tower_conv1_1 = batch_norm(tower_conv1_1, phase_train, 'Branch_1/Conv2d_1a_3x3/BatchNorm')
            tower_conv1_1 = tf1.nn.relu(tower_conv1_1)
            
            tower_conv2 = conv2d(net, 256, 1, scope='Branch_2/Conv2d_0a_1x1')
            tower_conv2 = batch_norm(tower_conv2, phase_train, 'Branch_2/Conv2d_0a_1x1/BatchNorm')
            tower_conv2 = tf1.nn.relu(tower_conv2)
            tower_conv2_1 = conv2d(tower_conv2, 256, 3, scope='Branch_2/Conv2d_0b_3x3')
            tower_conv2_1 = batch_norm(tower_conv2_1, phase_train, 'Branch_2/Conv2d_0b_3x3/BatchNorm')
            tower_conv2_1 = tf1.nn.relu(tower_conv2_1)
            tower_conv2_2 = conv2d(tower_conv2_1, 256, 3, stride=2, scope='Branch_2/Conv2d_1a_3x3', padding='VALID')
            tower_conv2_2 = batch_norm(tower_conv2_2, phase_train, 'Branch_2/Conv2d_1a_3x3/BatchNorm')
            tower_conv2_2 = tf1.nn.relu(tower_conv2_2)
            
            net = tf1.concat([tower_conv_1, tower_conv1_1, tower_conv2_2], 3)
        
        # 5x block8 (Inception-ResNet-C block)
        with tf1.variable_scope('Repeat_2'):
            for i in range(5):
                net = block8(net, 0.20, True, phase_train, f'Block8_{i+1}')
        
        # Final Inception-ResNet-C block
        net = block8(net, activation_fn=False, scope='Block8_6', phase_train=phase_train)
        
        # Global average pooling
        net = tf1.reduce_mean(net, [1, 2], keepdims=True, name='AvgPool')
        net = tf1.layers.dropout(net, rate=0.2, training=phase_train, name='Dropout')
        net = tf1.layers.flatten(net, name='Flatten')
        
        # Bottleneck layer
        with tf1.variable_scope('Bottleneck'):
            net = tf1.layers.flatten(net)
            net = tf1.matmul(
                net,
                tf1.get_variable(
                    'weights',
                    [net.get_shape()[-1], embedding_size],
                    initializer=tf1.glorot_uniform_initializer()
                )
            )
            net = tf1.nn.bias_add(
                net,
                tf1.get_variable(
                    'biases',
                    [embedding_size],
                    initializer=tf1.zeros_initializer()
                )
            )
            net = batch_norm(net, phase_train, 'BatchNorm')
        
        # Create embeddings node at root scope
        embeddings = tf1.identity(net, name='embeddings')
        return embeddings

def convert_model():
    """Convert DeepFace's Facenet512 model to TF1.x frozen graph"""
    
    # Enable eager execution temporarily to get weights
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    
    # Load the pre-trained model and get weights while in eager mode
    print("Loading DeepFace model...")
    model = DeepFace.build_model(model_name="Facenet512")
    keras_model = model.model
    weights = {}
    for layer in keras_model.layers:
        weights[layer.name] = layer.get_weights()
        print(f"Layer: {layer.name}, Weights shapes: {[w.shape for w in layer.get_weights()]}")
    
    # Reset TensorFlow graph and disable eager execution
    tf1.reset_default_graph()
    
    # Create new TF1.x graph
    with tf1.Graph().as_default() as graph:
        # Create placeholders
        inputs = tf1.placeholder(tf1.float32, [None, 160, 160, 3], name='input')
        phase_train = tf1.placeholder(tf1.bool, name='phase_train')
        
        # Create the network
        print("Creating TF1.x network...")
        net = create_inception_resnet_v1(inputs, phase_train)
        
        # Create embeddings node at root scope
        embeddings = tf1.identity(net, name='embeddings')
        
        # Create session and initialize variables
        config = tf1.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf1.Session(config=config) as sess:
            sess.run(tf1.global_variables_initializer())
            
            # Get all variables in our graph
            graph_vars = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES)
            print("\nTF1.x variables:")
            for var in graph_vars:
                print(f"{var.name}: {var.shape}")
            
            # Map and assign weights from Keras model to our TF1.x graph
            assign_ops = []
            for var in graph_vars:
                var_name = var.name.split(':')[0]
                # Remove InceptionResnetV1/ prefix for matching
                clean_name = var_name.replace('InceptionResnetV1/', '')
                
                for layer_name, layer_weights in weights.items():
                    if layer_name in clean_name:
                        if len(layer_weights) > 0 and layer_weights[0].shape == var.shape:
                            print(f"Assigning weights: {var_name} <- {layer_name}")
                            assign_ops.append(tf1.assign(var, layer_weights[0]))
                            break
            
            # Run assignments
            print("\nAssigning weights...")
            sess.run(assign_ops)
            
            # Save the frozen graph
            print("Freezing graph...")
            frozen_graph = tf1.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                ['embeddings']
            )
            
            # Remove any TF2.x attributes from the graph
            for node in frozen_graph.node:
                if 'explicit_paddings' in node.attr:
                    del node.attr['explicit_paddings']
                if 'grad_a' in node.attr:
                    del node.attr['grad_a']
                if 'grad_b' in node.attr:
                    del node.attr['grad_b']
            
            print("Saving frozen graph...")
            tf1.train.write_graph(
                frozen_graph,
                ".",
                "facenet512_model.pb",
                as_text=False
            )
            print("Model saved as facenet512_model.pb")

if __name__ == '__main__':
    convert_model() 
