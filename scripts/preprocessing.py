import os
import numpy as np
import pandas as pd
import collections

from tensorflow.keras.preprocessing.sequence import pad_sequences
from music21 import *

def read_midi(file, channel):
  midi = converter.parse(file)

  midi_partitions = instrument.partitionByInstrument(midi)
  if (midi_partitions == None):
    return
    
  print("Loading music from: ", file)
  notes = []

  for part in midi_partitions.parts:
    #print(str(part))
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
    #print(song)
    if (song):
      songs.append(song)
      i+=1

  print("Extracted "+channel+" channel from "+str(i)+" songs")
  return songs


def make_vocabs(songs, threshold):
  tot_notes = [x for elem in songs for x in elem]
  items, values = np.unique(tot_notes, return_counts=True)
  ind = np.argsort(values)
  notes = np.flip(items[ind])
  values = np.flip(values[ind])
  restricted = values[values>threshold]
  final_notes = notes[:len(restricted)]

  vocab_notes = {'<PAD>': 0 ,'<UNK>': 1}
  for note in final_notes:
      vocab_notes[note] = len(vocab_notes)
  inverted_vocab_notes = {v: k for k, v in vocab_notes.items()}
  return vocab_notes, inverted_vocab_notes, restricted


def load_csv_dataset(path):
  midi = []
  dataset = pd.read_csv(path)
  songs = list(dataset['Notes'])
  # print(songs[0])
  for song in songs:
      l = song.split(",")
      d = [s.strip(" '[]' ") for s in l]
      midi.append(d)
  return midi


def noteIndex(note,vocab):
  if (note not in vocab):
      return vocab["<UNK>"]
  else:
      return vocab[note]
    

def indexList(song, vocab):
    ls = []
    for n in song:
        ls.append(noteIndex(n,vocab))
    return ls


def int2onehot(index, vocab_dim):
  vec = [0] * vocab_dim
  vec[index] = 1
  return vec


def dataset_preprocessing(songs, vocab, seq_length):
    x = []
    y = []
    for song in songs:
      for i in range(0, len(song) - seq_length, 1):
        input_notes = song[i : i + seq_length]
        output_note = song[i + seq_length]
        x.append(indexList(input_notes, vocab))
        y.append(int2onehot(noteIndex(output_note, vocab), len(vocab)))
     
    x = pad_sequences(x, truncating='post', padding='post', maxlen = seq_length)
    
    x = np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0], x.shape[1], 1)
    
    return  x, y