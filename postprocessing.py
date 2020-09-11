import numpy as np

from music21 import *


def generate_song(model, vocab, song_length, train_seq, seq_length):
  start = np.random.randint(0, len(train_seq)-1)
  pattern = train_seq[start].tolist() 

  final_song = []
  for note_index in range(song_length):
    prediction_input = np.reshape(pattern, (1,seq_length,1))
    prediction = model.predict(prediction_input, verbose=0)
    predicted_index = np.argmax(prediction)
    if predicted_index==1:
      prediction = prediction[0][2:]
      predicted_index = np.argmax(prediction)+2
      print('Replaced <UNK> with: ', vocab[predicted_index])
    predicted_note = vocab[predicted_index]
    final_song.append(predicted_note)
    pattern.append([predicted_index])
    pattern = pattern[1:len(pattern)]

  return final_song


def create_midi(prediction_seq, channel, path, name):
  offset = 0
  output_notes = []
  for element in prediction_seq:
    #element is a chord
    if ('.' in element or element.isdigit()):
      notes_in_chord = element.split(".")
      notes = []
      for current_note in notes_in_chord:
        new_note = note.Note(int(current_note))
        new_note.storedInstrument = get_instrument(channel)
        notes.append(new_note)
      new_chord = chord.Chord(notes)
      new_chord.offset = offset
      output_notes.append(new_chord)

    #element is a note
    else:
      new_note = note.Note(element)
      new_note.offset = offset
      new_note.storedInstrument = get_instrument(channel)
      output_notes.append(new_note)
    
    offset += 0.5

  midi_stream = stream.Stream(output_notes)
  midi_stream.write('midi', fp=path+name)  



def get_instrument(channel):
  if (channel == "Piano"):
    return instrument.Piano()
  elif (channel == "Guitar"):
    return instrument.Guitar()
  elif (channel == "Bass"):
    return instrument.Bass()
  elif (channel == "Percussion"):
    return instrument.Percussion()
  else:
    return instrument.Piano()  