import cv2
import keras
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.nn import relu, softmax, dropout
from tensorflow.train import AdamOptimizer
from keras.datasets import cifar10

def normalize(t):
    max_value = np.max(t)
    min_value = np.min(t)
    return (t-min_value) / (max_value - min_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Choose one {train, test / predict}')
    parser.add_argument('--image_path', default=None, type=str, help='Path to image to predict')
    parser.add_argument('--learning_rate', default=0.00005, type=int, help='Learning rate for Adam optimizer.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train on.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch sizes for training.')
    parser.add_argument('--dropout', default=0.5, type=int, help='Dropout keep prob')
    parser.add_argument('--cnn_stack', default=5, type=int, help='Number of CNN layers to use in model')
    parser.add_argument('--fc_stack', default=3, type=int, help='Number of FC layers to use in model') 
    parser.add_argument('--fc1', default=512, type=int, help='Size of first hiddent FC layer')
    parser.add_argument('--pool_size', default=3, type=int, help='Size of pooling size for max pooling layer')
    args = parser.parse_args()
 
    if args.mode.lower() == 'train':    
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()       

        #Import data
        x_train = np.reshape(x_train, (x_train.shape[0], 32, 32, 3)) # 50000, 3072
        x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3)) # 10000, 3072
        x_train = normalize(x_train)
        x_test = normalize(x_test)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
       
        x_val, y_val = x_train[:5000], y_train[:5000]
        x_train, y_train = x_train[5000:], y_train[5000:]
         
        #Create the model
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x')
        y_ = tf.placeholder(tf.int64, shape=(None), name='y_')


        # Variables
        old_cnn_layer = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, padding='same')
        old_cnn_layer = tf.layers.batch_normalization(old_cnn_layer, momentum=0.9)
        old_cnn_layer = tf.nn.relu(old_cnn_layer)
        #old_cnn_layer = tf.layers.max_pooling2d(old_cnn_layer, pool_size=args.pool_size, strides=1, padding='same')

        for _ in range(args.cnn_stack-1):
            cnn_layer = tf.layers.conv2d(inputs=old_cnn_layer, filters=32, kernel_size=5, padding='same')
            cnn_layer = tf.layers.batch_normalization(cnn_layer, momentum=0.9)
            cnn_layer = tf.nn.relu(cnn_layer)
            #cnn_layer = tf.layers.max_pooling2d(old_cnn_layer, pool_size=args.pool_size, strides=1, padding='same')
            old_cnn_layer = cnn_layer
                 
        cnn_layer = tf.layers.flatten(cnn_layer)

        W = tf.get_variable('w', (32768, args.fc1),
                            initializer=tf.contrib.layers.xavier_initializer()) 
        b = tf.get_variable('b', (args.fc1,),
                            initializer=tf.contrib.layers.xavier_initializer()) 

        fc1 = args.fc1
        old_fc_layer = tf.matmul(cnn_layer, W) + b
        old_fc_layer = tf.layers.batch_normalization(old_fc_layer, momentum=0.9)
        old_fc_layer = tf.nn.relu(old_fc_layer) 

        if not args.fc_stack:
            fc_layer = old_fc_layer

        for idx in range(args.fc_stack-1):
            W = tf.get_variable('w'+str(idx+1), (fc1, fc1/2),
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b'+str(idx+1), (fc1/2,),
                                initializer=tf.contrib.layers.xavier_initializer())
        
            fc_layer = tf.matmul(old_fc_layer, W) + b
            fc_layer = tf.layers.batch_normalization(fc_layer, momentum=0.9)
            fc_layer = tf.nn.relu(fc_layer)
            old_fc_layer = fc_layer
            fc1 = fc1/2

        W = tf.get_variable('w'+str(args.fc_stack), (fc1, 10),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b'+str(args.fc_stack), (10,),
                            initializer=tf.contrib.layers.xavier_initializer())         
        y_pred = tf.add(tf.matmul(fc_layer, W),b, name='y_pred')

        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_,10),
                                                              logits=y_pred), name='loss')
        opt_min = tf.train.AdamOptimizer(learning_rate=0.0005, epsilon=0.000001, use_locking=True) \
                          .minimize(loss, name='Adam')
        
        pred = tf.cast(tf.nn.in_top_k(y_pred, y_, 1), tf.float32, name='prediction')
        accuracy = tf.reduce_mean(pred, name='accuracy')

        with tf.Session() as sess:
            # Training set = x_test
            # Val set = x_val
            sess.run(tf.initialize_all_variables())
            print('Loop\t\tTrain Loss\tTrain Acc %\tTest Loss\tTest Acc %')
            for e in range(args.epochs):
                idx_train = np.random.choice(45000, args.batch_size)
                idx_test = np.random.choice(5000, args.batch_size)
                X_batch_train, y_batch_train = x_train[idx_train], y_train[idx_train]
                X_batch_test, y_batch_test = x_test[idx_test], y_test[idx_test]
                _, train_loss, train_acc = sess.run([opt_min, loss, accuracy], 
                                                    feed_dict={x: X_batch_train, y_: y_batch_train})
                _, val_loss, val_acc = sess.run([opt_min, loss, accuracy], 
                                                  feed_dict={x: X_batch_test, y_: y_batch_test})
                print(f'{e}/{args.epochs}\t\t{val_loss:.4f}\t\t{val_acc*100:2.4f}\t\t{train_loss:.4f}\t\t{train_acc*100:2.4f}')

            # Test trained model
            _, test_loss, test_acc = sess.run([opt_min, loss, accuracy], feed_dict={x: x_test, y_: y_test})
            print(f'\nFinal Result\tTested Loss\tTested Acc %')
            print(f'\t\t{test_loss:.4f}\t\t{test_acc*100:2.4f}')
    
            saver = tf.train.Saver()
            saver.save(sess, 'model/baseline_model')        
            
    
    if args.mode.lower() == 'test' or args.mode.lower() == 'predict':
        if args.image_path == None:
            print('Invalid image path.')
        
        img = cv2.imread(args.image_path)
        img = np.reshape(img, (1, 3072)) # 1, 3072
        img = img / 255 
        img = img.astype(np.float32)

        x = tf.placeholder(tf.float32, shape=(None, 3072))
        y_ = tf.placeholder(tf.int64, shape=(None))

        saver = tf.train.import_meta_graph('model/baseline_model.meta')

        with tf.Session() as sess:
            saver.restore(sess,tf.train.latest_checkpoint('model/'))
            
            graph = tf.get_default_graph()
            y_pred = sess.run('y_pred:0', feed_dict={'x:0': img})

            loss = sess.run('loss:0', feed_dict={'y_pred:0': y_pred, 'y_:0': [3]})
        
            pred = sess.run('prediction:0', feed_dict={'y_pred:0': y_pred, 'y_:0': [3]})
            accuracy = sess.run('accuracy:0', feed_dict={'prediction:0': pred})
            print(f'Class\tLoss\tAcc')
            print(f'{y_pred.argmax()}\t{loss:.4f}\t{accuracy*100:2.4f}')
