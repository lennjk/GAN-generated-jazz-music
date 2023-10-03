# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 08:55:38 2021

@author: Lennart Kieschnik
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

generator = keras.models.load_model('generator_model_jazz.h5')

z_size=30             # Anzahl der Zufallszahlen zur Bildgeneration
                       # muss zum trainierten Modell passen

examples = 100
        
noise = np.random.normal(0, 1, size=[examples, z_size])


generated_data = generator.predict(noise)
print(generated_data.shape)

# PLOT TO VISUALIZE TO SUPPORT CHOOSING THE TRACK TO CONVERT TO MIDI

dim=(10, 10)
figsize=(10, 10)
plt.figure(figsize=figsize)
for i in range(generated_data.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(generated_data[i], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
    plt.tight_layout()
    
plt.show()
# ASK WHICH PLOT TO CONVERT

convert_dim = int(input("Which Pianoroll should be converted to MIDI? - "))

# WRITE RESULT INTO MIDI

# print(np.unique(generated_data[0, :, :, 0].astype("int")))
gen_jazz = np.where(generated_data[convert_dim, :, :, 0].astype("int")>-0.25,100,0)
print(gen_jazz.shape)
gen_jazz_dict = {"unnamed":gen_jazz}
import pdb; pdb.set_trace()
import write_midi
write_midi.write_midi(gen_jazz_dict, 4, "GAN_Jazz.mid", 80)
