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
    parser.add_argument('--learning_rate', default=0.001, type=int, help='Learning rate for Adam optimizer.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train on.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch sizes for training.')
    parser.add_argument('--dropout', default=0.5, type=int, help='Dropout keep prob')
    args = parser.parse_args()
 
    if args.mode.lower() == 'train':    
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()       

        #Import data
        x_train = np.reshape(x_train, (x_train.shape[0], -1)) # 50000, 3072
        x_test = np.reshape(x_test, (x_test.shape[0], -1)) # 10000, 3072
        x_train = normalize(x_train)
        x_test = normalize(x_test)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
       
        x_val, y_val = x_train[:5000], y_train[:5000]
        x_train, y_train = x_train[5000:], y_train[5000:]
         
        #Create the model
        x = tf.placeholder(tf.float32, shape=(None, 3072), name='x')
        y_ = tf.placeholder(tf.int64, shape=(None), name='y_')


        # Variables
        W1 = tf.get_variable('w1', (3072, 512),
                             initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable('w2', (512, 256), 
                             initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable('w3', (256, 128),
                             initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable('w4', (128, 64),
                             initializer=tf.contrib.layers.xavier_initializer())
        W5 = tf.get_variable('w5', (64, 10),
                             initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable('b1', (512,),
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', (256,),
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3', (128,),   
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable('b4', (64,),
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.get_variable('b5', (10,),
                             initializer=tf.contrib.layers.xavier_initializer())
        

        layer_1 = tf.matmul(x, W1) + b1
        layer_1 = tf.layers.batch_normalization(layer_1, momentum=0.9)
        layer_1 = tf.nn.relu(layer_1, name='layer_1')

        layer_2 = tf.matmul(layer_1, W2) + b2
        layer_2 = tf.layers.batch_normalization(layer_2, momentum=0.9)        
        layer_2 = tf.nn.relu(layer_2, name='layer_2')

        layer_3 = tf.matmul(layer_2, W3) + b3
        layer_3 = tf.layers.batch_normalization(layer_3, momentum=0.9)        
        layer_3 = tf.nn.relu(layer_3, name='layer_3')
        
        layer_4 = tf.matmul(layer_3, W4) + b4
        layer_4 = tf.layers.batch_normalization(layer_4, momentum=0.9)
        layer_4 = tf.nn.relu(layer_4, name='layer_4')

        y_pred = tf.add(tf.matmul(layer_4, W5),b5, name='y_pred')

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
                if e % (args.epochs/10) == 0:
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
