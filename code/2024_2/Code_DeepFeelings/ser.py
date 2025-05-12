# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:21:47 2024

@author: ernes
"""
import os
import cv2
import pandas as pd
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Ruta que deseas establecer como directorio de trabajo
ruta_trabajo = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/Corpus_Globalv1'
# Paso 1: Establecer la ruta de trabajo
os.chdir(ruta_trabajo)
# %%
path = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/Corpus_Globalv1'

# List all folders in the directory
folder_list = os.listdir(path)
print("Folders:",folder_list)
# %%
# Construct folder paths
folder_paths = [os.path.join(path, folder,"").replace("\\", "/") for folder in folder_list]
print(folder_paths)
#%%
# Get names of files within each folder
audio_extensions={".wav"}
file_names = []
for folder_path in folder_paths:
    if os.path.isdir(folder_path):  # Ensure it’s a folder
        files = [os.path.join(folder_path, f).replace("\\", "/") 
                 for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in audio_extensions
        ]
        file_names.extend(files)
# Display some information
print("First 6 file names:", file_names[:6])
print("Last 6 file names:", file_names[-6:])
print("Total number of files:", len(file_names))
#%%
import re
audio_folder= []
audio_number = []
valence = []
activation = []
dominance = []

for file in file_names:
    # Extract just the filename without directory path
    filename = os.path.basename(file) 
    # Use regex to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)

    # If there are exactly 5 numbers in the filename, append to respective arrays
    if len(numbers) == 5:
        audio_folder.append(int(numbers[0]))
        audio_number.append(int(numbers[1]))
        valence.append(int(numbers[2]))
        activation.append(int(numbers[3]))
        dominance.append(int(numbers[4]))
    else:
        # Optionally, handle cases where the number of numbers doesn't match expectation
        print(f"Warning: {filename} does not match expected number format.")

# Display the first few entries of each array to verify
print("First Number:", audio_folder[:5])
print("Second Number:", audio_number[:5])
print("Third Number:", valence[:5])
print("Fourth Number:", activation[:5])
print("Fifth Number:", dominance[:5])


#%%
y1,sr1=librosa.load(file_names[1])
S1=librosa.feature.melspectrogram(y=y1, sr=sr1,n_mels=256)
#%%


y, sr = librosa.load('./Audio2/Audio2_3-02-03-02.wav')
#y:datos de audio que eran procesados
#sr: convierte el tiempo a segundos o a una frecuencia especifica

#espectograma de mel
#%%
#n_mels: numero de bandas de mel a generar
sr=16000
S=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=256, hop_length=256)
print("Waveform Data (y):", y[:10])  # First 10 amplitude values
print("Sample Rate (sr):", sr)       # Sample rate in Hz

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,hop_length=256,
                         fmax=16000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
print(np.min(S_dB))
#%%
#preprocesamiento ///denoise

# Load audio file
file_path = './Audio4/Audio4_5-03-03-04.wav'
y, sr = librosa.load(file_path)

# Display an audio player
#pip install sounddevice
import sounddevice as sd
sd.play(y,sr)
#%%
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
import soundfile as sf
sr=16000
#loading audio
with AudioFile(file_path).resampled_to(sr) as f:
    audio = f.read(f.frames)
#noisereduction
reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
#enhancing through pedalboard
#board = Pedalboard([
#    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
#    Compressor(threshold_db=-16, ratio=4),
#    LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
#    Gain(gain_db=2)
#])

#effected = board(reduced_noise, sr)

#saving enhanced audio
path2 = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'
os.chdir(path2)
with AudioFile('./audio4_5.wav', 'w', sr, reduced_noise.shape[0]) as f:
  f.write(reduced_noise)
file_path_e = './audio_5_112.wav'

#%%
ye, sre = librosa.load(file_path_e)
sd.play(ye,sre)

Se=librosa.feature.melspectrogram(y=ye, sr=sre,n_mels=256)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(Se, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')


Se
mini=np.max(Se)
maxi=np.min(Se)
#x_min/max-min
#min max scaler
arr2=(Se-mini)/(maxi-mini)
arr2
np.max(arr2)
np.min(arr2)
#para vovver a original, min max scaler -1



#%%
#preprocesamiento

#pip install soundfile

# Load the audio file
input_file = './Audio2/Audio2_3-02-03-02.wav'
y, sr = librosa.load(input_file, sr=None)  # sr=None to maintain original sampling rate

# Define FIR filter coefficients from the given formula
# H(z) = 1 - az^-1, where a = -0.97
a = -0.97
fir_coeffs = [1, a]  # The filter coefficients, corresponding to 1 and -0.97

# Apply the filter using convolution
filtered_audio = np.convolve(y, fir_coeffs, mode='same')
path2 = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo'

# Write the filtered audio to a new WAV file
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'
librosa.output.write_wav(output_file, filtered_audio, sr)
librosa.output.write_wav
print(f"FIR filtered audio saved to {output_file}")





#%%

#leer archivo 
## fn_wav = os.path.join('..', 'data', 'B', 'FMP_B_Note-C4_Piano.wav')

###
#integrar egemapas
#despues de flatten
#add data after flattening

#%%
#generar nombre de spec
x=audio_folder[3]
y=audio_number[3]
name=f"spec_{x}_{y}.png"
print(name)
output_path = os.path.join(output_file, name).replace("\\", "/")
print(output_path)
len(file_names)
#%%
#generar sepectogrmamas no filtradps
sr=16000
i=0
os.chdir(path)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mel_spec'
os.makedirs(output_file, exist_ok=True)
for i in range(len(file_names)):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"spec_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    S=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=256, hop_length=256)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr,hop_length=256,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.savefig(output_path)
    plt.close()
    print(i)
    

#%%
#generar csv
import csv
data=list(zip(audio_folder,audio_number,valence,activation,dominance))
ruta=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/vad.csv'
with open(ruta,"w",newline="") as file:
    writer=csv.writer(file)
    writer.writerow(["audio_folder","audio_number","valence","activation","dominance"])
    writer.writerows(data)
#%%
#generar audios filtrados
i=0
os.chdir(path)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mel_spec'
os.makedirs(output_file, exist_ok=True)
for i in range(len(file_names)):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"spec_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    S=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=256, hop_length=256)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr,hop_length=256,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.savefig(output_path)
    plt.close()
    print(np.min(S_dB))