# Script for the analysis of the datasets.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from music21 import *
import os


def read_midi(file, channel):
  midi = converter.parse(file)

  midi_partitions = instrument.partitionByInstrument(midi)
  if (midi_partitions == None):
    return
    
  print("Loading music from: ", file)
  notes = []

  for part in midi_partitions.parts:
    if (channel in str(part)):
      note_iterator = part.recurse()
      for note_ in note_iterator:
        if (isinstance(note_, note.Note)):
          notes.append(str(note_.pitch))
        elif (isinstance(note_, chord.Chord)):
          notes.append('.'.join(str(n) for n in note_.normalOrder))
  return notes

def load_dataset(path, channel="Piano"):
  songs = []
  i = 0
  for file in os.listdir(path):
    song = read_midi(path+file, channel)
    if (song):
      songs.append(song)
      i+=1

  print("Extracted "+channel+" channel from "+str(i)+" songs")
  songs_ = [x for elem in songs for x in elem]
  return songs_


# --- Function that extracts the list of all the notes of all the songs in the Kaggle Dataset. ---
# @params path: path to the .csv file of the Kaggle dataset
# @return tot_notes: list of all the notes occurred in the dataset (repetitive presence)

def get_data_kaggle(path):
    midi = []

    dataset = pd.read_csv(path) 
    songs = list(dataset['Notes'])
    for song in songs:
        l = song.split(",")
        d = [s.strip(" '[]' ") for s in l]
        midi.append(d)
    tot_notes = [x for elem in midi for x in elem]
    return tot_notes


def get_data_novel():
    return

# --- Function that sorts the elements of a list with respect to their occurrency count. ---
# @param list_: list of repeated notes
# @param relative: whether frequencies have to be relative (default True)
# @return items_sorted: list of unique notes in increasing order of occurrences
# @return values_sorted: list of frequency of unique notes in increasing order of occurrences

def get_sorted(list_, relative=True):

    items, values = np.unique(list_, return_counts=True)
    items_sorted = [x for _,x in sorted(zip(values,items))]
    values_sorted = [x for _,x in sorted(zip(values,values))]
    if relative:
        values_sorted = np.asarray(list(values_sorted))
        tot_sum = np.sum(values_sorted)
        values_sorted = (values_sorted/tot_sum)*100

    return np.asarray(list(items_sorted)), values_sorted


# --- Function that plots the info about the distribution of notes. ---
# @params x: notes
# @params y: relative frequencies
# @return None: plots the distribution

def plot_info(x, y, relative=True):

    fig, ax = plt.subplots()
    ax.plot(x, y, '.')
    ax.set_xticks([])
    plt.xlabel('Notes')
    if relative:
        plt.ylabel('Relative Frequency (%)')
    else:
        plt.ylabel('Absolute Frequency')
    plt.show()


# --- Function that cuts the array of notes if their occurrency is below the threshold. ---
# @param x: numpy array of notes
# @param y: numpy array of frequencies
# @param threshold: threshold for filtering notes
# @return supp_x: numpy array of filtered notes
# @return tot_left: percentage of dataset left (ONLY FOR RELATIVE FREQUENCIES)
# @return size_vocab: number of unique notes left

def cut(x, y, threshold):

    supp_x = []
    supp_y = []
    for count, elem in enumerate(y):
        if elem > threshold:
            supp_x.append(x[count])
            supp_y.append(elem)
    supp_x = np.asarray(supp_x)
    supp_y = np.asarray(supp_y)
    tot_left = np.sum(supp_y)
    size_vocab = len(supp_x)
    return np.asarray(supp_x), [tot_left, size_vocab]
            


if __name__ == "__main__":
    
    threshold = 0.0
    path_kaggle = "./Jazz-midi.csv"
    path_novel = "../JazzDataset/"
    channel="Piano"

    #tot_notes = load_dataset(path_novel, channel)
    tot_notes = get_data_kaggle(path_kaggle)
    notes, counts = get_sorted(tot_notes)
    steps = np.arange(0, 100, 100/len(notes))
    red_notes, stats = cut(notes, counts, threshold)
    print('Dataset percentage left: ', stats[0])
    print('Size output: ', stats[1])
    plot_info(steps, counts)
