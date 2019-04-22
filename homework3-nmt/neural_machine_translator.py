import os
import time
import keras
import argparse
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from preprocess import preprocess
from bahdanau_attention import BahdanauAttention


def loss_function(real, pred):
    '''
        From TensorFlow NMT tutorial
    '''
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.metrics.sparse_categorical_crossentropy(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def nmt(input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, targ_lang):
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        
        example_input_batch, example_target_batch = next(iter(dataset))
        #example_input_batch, example_target_batch = dataset.make_initializable_iterator().get_next()

        # Encoder
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
        #print (f'Encoder output shape: (batch size, sequence length, units): {sample_output.shape}')
        #print (f'Encoder Hidden state shape: (batch size, units): {sample_hidden.shape}')

        # Attention layer
        attention_layer = BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        #print(f'Attention result shape: (batch size, units): {attention_result.shape}')
        #print(f'Attention weights shape: (batch_size, sequence_length, 1): {attention_weights.shape}')    

        # Decoder
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                          sample_hidden, sample_output)
        #print (f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')
        
        optimizer = tf.keras.optimizers.Adam()
        #loss_object = tf.keras.metrics.sparse_categorical_crossentropy
        #loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        #checkpoint_dir = './training_checkpoints'
        #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        #checkpoint = tf.train.Checkpoint(optimizer=optimizer,
        #                                 encoder=encoder,
        #                                 decoder=decoder)
        
        EPOCHS = 10

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, optimizer, BATCH_SIZE)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            #if (epoch + 1) % 2 == 0:
            #    checkpoint.save(file_prefix = checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, optimizer, BATCH_SIZE):
    '''
        From TensorFlow NMT tutorial
    '''

    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

if __name__ == '__main__':
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/')
    parser.add_argument('--num_examples', type=int, default=30000)
    parser.add_argument('--id_gpu', type=int)
    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.id_gpu
    
    print('Processing and loading data...')
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, target_lang = preprocess(args.data_path, args.num_examples)
    
    nmt(input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, target_lang)
