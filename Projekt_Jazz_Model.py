# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:20:54 2021

@author: Lennart Kieschnik
"""

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialisierung des Zufallszahlengenerators
np.random.seed(0)

# load previously created data
data = np.load('jazz_data.npy')

# Normalisierung auf -1 bis +1
data = data.astype('float32')  / max(np.unique(data)) * 2 -1
# print(np.unique(data))

# Pixel in separate Merkmale aufteilen
#real_images_all = real_images_all.reshape(-1, 784)
data = data.reshape(-1, 200, 128, 1)

# Anzahl der Zufallszahlen zur Erzeugung eines Bildes
z_size = 30 #64

# Optimizer festlegen
optimizer= RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)

# dropout ist wichtig bei GAN
dropout_rate = 0.2

# die Steigung der LeakyReLu im negativen Bereich
leaky_faktor = 0.2

# Neuronales Netz für Generator
from tensorflow.keras.layers import Conv2DTranspose

generator = Sequential()
generator.add(Dense(25*8*128, input_shape=[z_size]))
generator.add(Reshape([25, 8, 128]))
generator.add(BatchNormalization())
generator.add(LeakyReLU(leaky_faktor))
generator.add(Conv2DTranspose(64, kernel_size=5, strides=(2,4), padding='same',
                              activation='linear'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(128, kernel_size=5, strides=(2,2), padding='same',
                              activation='linear'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(leaky_faktor))
generator.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding='same',
                              activation='tanh'))

#generator.compile(loss='binary_crossentropy', optimizer=optimizer)
generator.summary()

# Neuronales Netz für Diskriminator
discriminator = Sequential()
discriminator.add(Conv2D(32,(5,5),strides=(1,1), padding='same',
                      activation='relu',input_shape=(200,128,1)))
discriminator.add(MaxPool2D(pool_size=(2, 2),padding='same'))
discriminator.add(Conv2D(32,(3,3),padding='same',activation='relu'))
discriminator.add(MaxPool2D(pool_size=(2, 2),padding='valid'))
discriminator.add(Conv2D(64,(3,3),padding='same', activation='relu'))
discriminator.add(MaxPool2D(pool_size=(2, 2),padding='same'))
discriminator.add(Conv2D(64,(3,3),padding='same',activation='relu'))
discriminator.add(MaxPool2D(pool_size=(2, 2),padding='same'))
discriminator.add(Flatten())
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(200,activation='relu'))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(100,activation='relu'))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.summary()

discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')


# der Diskriminator wird hier zunächst eingefroren
discriminator.trainable = False 

# Dummy Input Vektor
z = Input(shape=(z_size,))
print(z)

# Durch die Model-Funktion werden Generator und Deskriminiator verkettet
gan_core = discriminator(generator(z))
gan = Model(inputs = z, outputs = gan_core)
gan.compile(loss = 'binary_crossentropy', optimizer = optimizer)
gan.summary()


epochs = 90
batch_size = 100
batch_count = data.shape[0] / batch_size

history_generator=[]
history_discriminator=[]

for e in range(1, epochs+1):
    print('-'*15, 'Epoche %d' % e, '-'*15)
    for count in tqdm(range(int(batch_count))):
        # Zufallszahlenvektor erzeugen
        z = np.random.normal(0, 1, size=[batch_size, z_size])
        # Fakebilder erzeugen
        fake_data = generator.predict(z)
        
        # Reale Biler abrufen un mit Fakebildern zusammenfügen
        real_data = data[np.random.randint(0, 
                        data.shape[0], size=batch_size)]
        x_dis = np.concatenate([real_data, fake_data])
        # Labels erzeugen: 0 für reale Bilder 1 für Fakebilder
        y_dis = np.zeros(2*batch_size)
        y_dis[batch_size:] = 1
        
        # Diskriminator einschalten und auf dem erzeugten Datensatz fitten
        discriminator.trainable = True
        history_disc=discriminator.fit(x_dis, y_dis, verbose=False, batch_size=2*batch_size)
        discriminator.trainable = False        
        history_discriminator.append(history_disc.history['loss'])
        
        # Als nächstes noch den Generator fitten
        #x_gan = np.random.normal(0, 1, size=[batch_size, z_size])
        y_gan = np.zeros(batch_size)
        history_gen=gan.fit(z, y_gan, verbose=False, batch_size=batch_size)
        history_generator.append(history_gen.history['loss'])
        
  
    if e == 1 or e==3 or e==5 or e % 10 == 0:
        
       
        examples = 100
        dim=(10, 10)
        figsize=(10, 10)
        noise = np.random.normal(0, 1, size=[examples, z_size])
        generated_data = generator.predict(noise)
        generated_data = generated_data.reshape(examples, 200, 128)
    
        # Demobilder anzeigen
        plt.figure(figsize=figsize)
        for i in range(generated_data.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_data[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'learning_progress_{e}.png')
        plt.show()
        plt.close()
    
        generator.save('generator_model_jazz.h5')
        
        plt.plot(history_discriminator, label='Discriminator loss')
        plt.plot(history_generator, label='Generator loss')
        plt.legend()
        plt.show()