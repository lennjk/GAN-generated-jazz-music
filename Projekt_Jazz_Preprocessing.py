# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:46:57 2021

@author: Lennart Kieschnik
"""

import read_midi
import write_midi
import numpy as np
import os

# PREPROCESSING

# Load midi files from directory by iterating the directory and applying a 
# customized function

data = []

directory = "C://Users//Lennart Kieschnik//DL//Projekt//midi//Jazz Midi//"
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # print(os.path.join(directory, filename))
    try:
        midi_dict = read_midi.Read_midi(directory+filename, 4).read_file()
        midi_array = np.array([midi_dict[i] for i in list(midi_dict.keys())])
         
        list_of_note_counts = []
        for i in range(midi_array.shape[0]):
            counter = (midi_array[i,:,:] > 0).sum()
            list_of_note_counts.append(counter)
        
        # this function finds the area of highest note density and reduces the
        # array to it, making all music pieces uniform in size
        def get_heighest_zone(arr, zone_size_row, zone_size_col):
            max_sum = float("-inf")
            row_idx, col_idx = 0, 0
            for row in range(arr.shape[0]-zone_size_row):
                for col in range(arr.shape[1]-zone_size_col):
                    curr_sum =  np.sum(arr[row:row+zone_size_row, col:col+zone_size_col])
                    if curr_sum > max_sum:
                        row_idx, col_idx = row, col
                        max_sum = curr_sum
            return arr[row_idx:row_idx+zone_size_row, col_idx:col_idx+zone_size_col]
        
        midi_array_density = get_heighest_zone(midi_array[list_of_note_counts.index(max(list_of_note_counts)),:,:], 200, 128)
        midi_array_density = np.where(midi_array_density > 0, 100, 0)
        
        # this if command excludes music pieces that are too short
        if midi_array_density.shape[0] < 200:
            continue
        # print(midi_array_density)     
        data.append(midi_array_density)
    except:
        continue

data = np.array(data)
print(data, data.shape)
np.save('jazz_data', data)
r"""
filepath = r"C:\Users\Lennart Kieschnik\DL\Projekt\midi\Jazz Midi\Love.mid"
aaa = read_midi.Read_midi(filepath, 4).read_file()
    
bbb = np.array([aaa[i] for i in list(aaa.keys())])
print(bbb, bbb.shape)
print(np.unique(bbb))
"""