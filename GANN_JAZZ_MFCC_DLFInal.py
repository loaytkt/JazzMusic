#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from google.colab import drive
drive.mount("/content/drive")


path = '/content/Music'

#Install the stft module o colab
get_ipython().system('pip install librosa')


# In[ ]:


# installing libs
import tensorflow as tf
import glob
import numpy as np
import os
import glob
import struct
import wave
import random

import librosa
import matplotlib.pyplot as plt
import soundfile

flag = True


# In[ ]:



# data preparation
def generate_mfcc(path):
    # getting the file data
    data, sr = librosa.load(path)
    # converting it to mfcc with dct
    mfcc = librosa.feature.mfcc(y=data, sr=16000, dct_type = 2)

    return mfcc, sr

# plotting the MFCC
def plot_chroma(chroma):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')

    # loading the data as an array
def load_wavfile_as_array(input_list):
    mfccs = []
    srs=[]
    
    for input_file in input_list:
        mfcc, sr = generate_mfcc(input_file)
  
        mfccs.append(mfcc)
        srs.append(sr)
    
    return np.asarray(mfccs), np.asarray(sr)


# In[ ]:


mfcc_list = []
sr_list = []

# going through the audio files and saving their mfcc in arrays, sr is the sample rate
for dir in os.listdir(path):
    
    if(dir != ".ipynb_checkpoints"):
        input_list = glob.glob(path+'/'+ dir +'/*.wav')
        mfccs, sr= load_wavfile_as_array(input_list)

    mfcc_list.append(mfccs)
    sr_list.append(sr)

print('Data Loaded!')


# In[ ]:


BUFFER_SIZE = 60000 #60000
BATCH_SIZE = 4    #64
EPOCHS = 10000   #100


# creating a training data set from the dataset that was inputted
train_dataset = tf.data.Dataset.from_tensor_slices(mfcc_list[0]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_dataset)


# In[ ]:


# creating the generator model
def make_generator_model():
    model = tf.keras.Sequential([
        # 20 x 130 is the input layer. 100 is the noise dimension
        tf.keras.layers.Dense(20*130, input_shape = (100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((20, 130)),
        tf.keras.layers.Conv1DTranspose(256,7, padding = "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1DTranspose(64,7, strides=2, padding = "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1DTranspose(2,7, strides=2, padding = "same"),

        tf.keras.layers.Flatten(),


        tf.keras.layers.Dense(20*130),
    ])

    return model

generator = make_generator_model()


# In[ ]:


def make_discriminator_model():
    model = tf.keras.Sequential([

      
          
          tf.keras.layers.Reshape((20, 130)),

          tf.keras.layers.Conv1D(512, 7, padding='same', name='1Conv1D', dtype='float32'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dropout(0.3),

          tf.keras.layers.Dense(50, name='dense1', activation = "relu"),

          tf.keras.layers.Conv1D(256, 7, padding='same'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dropout(0.3),

          tf.keras.layers.Conv1D(124, 7, padding='same'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dropout(0.3),

          #tf.keras.layers.Reshape((reduced_, 10000)),
          tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid"),
          tf.keras.layers.Flatten(),

          tf.keras.layers.Dense(2, name='dense10')
        ])

    return model

discriminator = make_discriminator_model()


# In[ ]:


# cross entropy function used to calculate the loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# definiing the desc loss
def loss_desc(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
# defining the gen_loss
def loss_gen(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#using optimizers to speed up the training
generator_optimizer = tf.keras.optimizers.Adam(1e-2)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-2)


# In[ ]:


num_examples_to_generate = 1 #50
noise_dim = 100
# creating a seed for the geneator
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# In[ ]:


def train_step(audio):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    # get the losses of the discriminator and gen in eahc step
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_audio = generator(noise, training=True)
        real_output = discriminator(audio, training=True)
        fake_output = discriminator(generated_audio, training=True)

        gen_loss = loss_gen(fake_output)
        disc_loss = loss_desc(real_output, fake_output)

    #Update weights of the two net based on the previous results
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[ ]:


def train(dataset, epochs):
    print('Training...')
    for epoch in range(epochs):

    for audio_batch in dataset:
        train_step(audio_batch)
    
    # Generate after the final epoch
    generate_and_save_audio(generator, epoch, seed)


# In[ ]:


def generate_and_save_audio(model, epoch, test_input):

    print ('Time for epoch {}'.format(epoch + 1))
    if ((epoch+1)%50 == 0):

        predictions = model(test_input, training=False)

    #plot_chroma(predictions)

        output = librosa.feature.inverse.mfcc_to_audio(np.asarray(predictions, dtype='float32'))

        # generate a .wav file if the eopch is multiple of 500
        if(epoch % 500 == 0):
            with wave.open('track'+str(epoch+1)+'.wav', 'w') as track_output:
                track_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
        
                track_output.writeframes(output)

   

  return


# In[ ]:


train(train_dataset, EPOCHS)

