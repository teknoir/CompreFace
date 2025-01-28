import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
from deepface import DeepFace
import numpy as np

def batch_norm(inputs, phase_train, scope):
    """Batch normalization with control dependencies"""
    return tf1.layers.batch_normalization(
        inputs=inputs,
        training=phase_train,
        name=scope,
        fused=True
    )

def conv2d(inputs, filters, kernel_size, stride=1, scope=None, padding='SAME'):
    """Convolution with batch norm and relu"""
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
        net = tf1.add(net, scale * up)
        net = tf1.nn.relu(net)
        return net

def create_inception_resnet_v1(inputs, phase_train=True, embedding_size=512):
    """Creates InceptionResNetV1 architecture with exact naming from davidsandberg"""
    
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
        
        net = tf1.layers.max_pooling2d(net, 3, 2, padding='VALID', name='MaxPool_3a_3x3')
        
        net = conv2d(net, 80, 1, scope='Conv2d_3b_1x1')
        net = batch_norm(net, phase_train, 'Conv2d_3b_1x1/BatchNorm')
        net = tf1.nn.relu(net)
        
        net = conv2d(net, 192, 3, scope='Conv2d_4a_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_4a_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        net = conv2d(net, 256, 3, stride=2, scope='Conv2d_4b_3x3', padding='VALID')
        net = batch_norm(net, phase_train, 'Conv2d_4b_3x3/BatchNorm')
        net = tf1.nn.relu(net)
        
        # 5x Inception-ResNet-A blocks
        with tf1.variable_scope('Repeat'):
            for i in range(5):
                net = block35(net, 0.17, phase_train, f'block35_{i+1}')
        
        # Reduction-A
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
            
            tower_pool = tf1.layers.max_pooling2d(net, 3, 2, padding='VALID', name='Branch_2/MaxPool_1a_3x3')
            net = tf1.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        
        # Bottleneck layer
        with tf1.variable_scope('Bottleneck'):
            net = tf1.layers.flatten(net)
            net = tf1.layers.dense(net, embedding_size, name='MatMul')
            net = batch_norm(net, phase_train, 'BatchNorm')
        
        # Move embeddings identity outside of all scopes
        net = tf1.identity(net)
    
    # Create embeddings node at root scope
    embeddings = tf1.identity(net, name='embeddings')
    return embeddings

def convert_model():
    """Convert DeepFace's Facenet512 model to TF1.x frozen graph"""
    
    # Enable eager execution temporarily to get weights
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    
    # Load the pre-trained model and get weights while in eager mode
    model = DeepFace.build_model(model_name="Facenet512")
    keras_model = model.model
    weights = {}
    for layer in keras_model.layers:
        weights[layer.name] = layer.get_weights()
    
    # Reset TensorFlow graph and disable eager execution
    tf1.reset_default_graph()
    
    # Create new TF1.x graph
    with tf1.Graph().as_default() as graph:
        # Create placeholders
        inputs = tf1.placeholder(tf1.float32, [None, 160, 160, 3], name='input')
        phase_train = tf1.placeholder(tf1.bool, name='phase_train')
        
        # Create the network
        embeddings = create_inception_resnet_v1(inputs, phase_train)
        
        # Debug: Print all node names
        print("\nAll node names in graph:")
        for n in graph.as_graph_def().node:
            print(n.name)
        
        # Create session and initialize variables
        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            
            # Get all variables in our graph
            graph_vars = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES)
            
            # Map and assign weights from Keras model to our TF1.x graph
            for var in graph_vars:
                var_name = var.name.split(':')[0]
                if var_name in weights:
                    sess.run(var.assign(weights[var_name]))
            
            # Save the frozen graph
            graph_def = sess.graph.as_graph_def()
            frozen_graph = tf1.graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['embeddings']  # Make sure this matches the actual node name
            )
            
            tf1.train.write_graph(
                frozen_graph,
                ".",
                "facenet512_model.pb",
                as_text=False
            )

if __name__ == '__main__':
    convert_model()