# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:11:06 2024

@author: ernes
"""
import os
import cv2
import pandas as pd
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import csv

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
path_f='C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'

# Get all file names in the directory
file_names2 = [f for f in os.listdir(path_f) if os.path.isfile(os.path.join(path_f, f))]

# Print the file names
for file_name in file_names2:
    print(file_name)
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




# a partir de aqui se corren celdas segun su uso 

#%% generar audios filtrados
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
import soundfile as sf
sr=16000
i=0
os.chdir(path)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'
os.makedirs(output_file, exist_ok=True)
for i in range(len(file_names)):
    a=audio_folder[i]
    b=audio_number[i]
    c=valence[i]
    d=activation[i]
    e=dominance[i]
    name=f"audio_{a}_{b}_{c}_{d}_{e}.wav"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    with AudioFile(file_names[i]).resampled_to(sr) as f:
        audio = f.read(f.frames)
        #noisereduction
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
        #enhancing through pedalboard
        board = Pedalboard([
            NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
            Compressor(threshold_db=-16, ratio=4),
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
            Gain(gain_db=2)
            ])
        effected = board(reduced_noise, sr)
        path2 = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'
        os.chdir(path2)
        with AudioFile(name, 'w', sr, effected.shape[0]) as f:
          f.write(effected)
        file_path_e = output_path
        print(i)


#%%
#generar sepectogrmamas filtrados
path_f='C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'

# Get all file names in the directory
file_names2 = [f for f in os.listdir(path_f) if os.path.isfile(os.path.join(path_f, f))]

# Print the file names
for file_name in file_names2:
    print(file_name)

audio_folder2= []
audio_number2 = []
valence2 = []
activation2 = []
dominance2 = []

for file in file_names2:
    # Extract just the filename without directory path
    filename = os.path.basename(file) 
    # Use regex to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)

    # If there are exactly 5 numbers in the filename, append to respective arrays
    if len(numbers) == 5:
        audio_folder2.append(int(numbers[0]))
        audio_number2.append(int(numbers[1]))
        valence2.append(int(numbers[2]))
        activation2.append(int(numbers[3]))
        dominance2.append(int(numbers[4]))
#%%
#generar espectograma filtrado
sr=16000
i=0
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/filter_spec'
os.makedirs(output_file, exist_ok=True)
for i in range(0, len(file_names), 5):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"spec_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names2[i],sr=sr)
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
#generar vaweplot
path_f='C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/audio_filtrado'


sr=16000
i=0
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/wave'
os.makedirs(output_file, exist_ok=True)
plt.figure(figsize=(15,6))
for i in range(0, len(file_names), 5):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"wave_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    img = librosa.display.waveshow(y, x_axis='time',sr=sr)
    ax.set(title='waveplot')
    plt.savefig(output_path)
    plt.close()
    print(i)

#%%
#generar waveplot filtrado

sr=16000
i=0
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/wave_filter'
os.makedirs(output_file, exist_ok=True)
plt.figure(figsize=(15,6))
for i in range(0, len(file_names2), 5):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"wave_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    img = librosa.display.waveshow(y, x_axis='time',sr=sr)
    ax.set(title='waveplot')
    plt.savefig(output_path)
    plt.close()
    print(i)
    
    
#%%
y,sr=librosa.load(file_names[11],sr=sr)
z= librosa.feature.zero_crossing_rate(y)
np.mean(z)
#%%

y,sr=librosa.load(file_names[11],sr=sr)
c= librosa.feature.spectral_centroid(y=y)
cc=np.mean(c)
cc
#%%
rms=librosa.feature.rms(y=y)
np.mean(rms)
#%%
#generar csv con zero crossing rate, spectral centroid, y root mean square
sr=16000
i=0
zm=[]
cm=[]
rmsm=[]
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/feat1'
os.makedirs(output_file, exist_ok=True)
plt.figure(figsize=(15,6))
for i in range(0, len(file_names)):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"wave_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    z= librosa.feature.zero_crossing_rate(y)
    zz=np.mean(z)
    zm.append(zz)
    c= librosa.feature.spectral_centroid(y=y)
    cc=np.mean(c)
    cm.append(cc)
    rms=librosa.feature.rms(y=y)
    rrms=np.mean(rms)
    rmsm.append(rrms)
    print(i)

data=list(zip(zm,cm,rmsm))
ruta=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/feats1.csv'
with open(ruta,"w",newline="") as file:
    writer=csv.writer(file)
    writer.writerow(["zero_cr_rate","centroide","root_mean_square"])
    writer.writerows(data)
#%%

sr=16000
i=0
zm=[]
cm=[]
rmsm=[]
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/feat1'
for i in range(0, len(file_names2)):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"wave_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names2[i],sr=sr)
    z= librosa.feature.zero_crossing_rate(y)
    zz=np.mean(z)
    zm.append(zz)
    c= librosa.feature.spectral_centroid(y=y)
    cc=np.mean(c)
    cm.append(cc)
    rms=librosa.feature.rms(y=y)
    rrms=np.mean(rms)
    rmsm.append(rrms)

    
    
#%%
#mfcc
sr=16000
hop_length=256
n_fft=1024
y1,sr=librosa.load(file_names[10],sr=sr)
y2,sr=librosa.load(file_names[1000],sr=sr)

mfccs = librosa.feature.mfcc(y=y1, sr=sr,hop_length=256,n_fft=1024)
print(mfccs.shape)
mfccs = librosa.feature.mfcc(y=y2, sr=sr,hop_length=256,n_fft=1024)
print(mfccs.shape)
#%%
#generar mfcc audios originales
import librosa
import numpy as np
import sklearn
from scipy.signal import resample

# Lista de rutas de los archivos de audio
audio_files = file_names

# Parámetros para la extracción de MFCCs
n_mfcc = 21
n_fft = 1024 #tamaño de ventana para la transformada de fourier
hop_length = 512 #numero de muestras que se saslta entre fotogramas consecutivos 
sr = 16000          # Frecuencia de muestreo
target_columns = 128  # Número deseado de columnas en la dimensión temporal

# Lista para almacenar los MFCCs interpolados
mfcc_list = []

for file in audio_files:
    # Cargar el archivo de audio
    y, _ = librosa.load(file, sr=sr)
    
    # Extraer los MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    #Escalamos caracterisiticas para que tengan media 0 y varianza 1
    # dejándo con 20 coeficientes en lugar de 21
    mfcc = mfcc[1:, :]
    mfcc=sklearn.preprocessing.minmax_scale(mfcc, axis=1)
    
    # Interpolación (resample) para que la dimensión de tiempo (columnas) sea fija
    mfcc_interpolated = resample(mfcc, target_columns, axis=1)
    
    # Agregar a la lista
    mfcc_list.append(mfcc_interpolated)

# Convertir a un arreglo numpy
data = np.array(mfcc_list)

# Mostrar las dimensiones finales
print(f"Dimensiones finales de los MFCCs ajustados: {data.shape}")


#%%
#generar mfcc audios filtrados
# Definir la ruta de la carpeta que quieres inspeccionar
ruta_carpeta = path_f

# Listar los archivos en la carpeta
archivos = os.listdir(ruta_carpeta)
rutas2=[]
# Iterar sobre los archivos y mostrar su nombre y ruta completa
for archivo in archivos:
    # Obtener la ruta completa del archivo
    ruta_completa = os.path.join(ruta_carpeta, archivo)
    rutas2.append(ruta_completa)

# Lista de rutas de los archivos de audio
audio_files = rutas2
# Parámetros para la extracción de MFCCs
n_mfcc = 21
n_fft = 1024
hop_length = 512
sr = 16000  # Frecuencia de muestreo fija
target_columns = 128  # Número deseado de columnas

# Procesar cada archivo de audio
mfcc_list2 = []

for file in audio_files:
    # Cargar el archivo de audio
    y, _ = librosa.load(file, sr=sr)
    
    # Extraer los MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    #Escalamos caracterisiticas para que tengan media 0 y varianza 1
    # se elimina el primer coeficiente
    mfcc = mfcc[1:, :]
    mfcc=sklearn.preprocessing.minmax_scale(mfcc, axis=1)
    # Interpolación (resample) para que la dimensión de tiempo (columnas) sea fija
    mfcc_interpolated = resample(mfcc, target_columns, axis=1)
    
    # Agregar a la lista
    mfcc_list2.append(mfcc_interpolated)

# Convertir a un arreglo para entrenamiento
data2 = np.array(mfcc_list2)

# Mostrar las dimensiones finales
print(f"Dimensiones finales de los MFCCs ajustados: {data.shape}")  # Salida: (n_audios, 13, 60)


#%%
ruta=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc1.csv'
with open(ruta,"w",newline="") as file:
    writer=csv.writer(file)
    writer.writerows(mfcc_list)

ruta=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc_filter.csv'
with open(ruta,"w",newline="") as file:
    writer=csv.writer(file)
    writer.writerows(mfcc_list2)

#%%
# Crear un DataFrame de pandas con esta matriz
data_2d = data.reshape(data.shape[0], -1)  # 3764 filas, 720 columnas
df = pd.DataFrame(data_2d)
ruta=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc1.csv'
df.to_csv(ruta, index=False)

#%%
# Crear un DataFrame de pandas con esta matriz
data2_2d = data2.reshape(data2.shape[0], -1)  # 3764 filas, 720 columnas
df2 = pd.DataFrame(data2_2d)
ruta2=r'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc1_filter.csv'
df2.to_csv(ruta2, index=False)

#%%imagenes mfcc
import sklearn
sr=16000
i=0
os.chdir(path_f)
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc_plot'
os.makedirs(output_file, exist_ok=True)
for i in range(0, len(file_names), 50):
    a=audio_folder[i]
    b=audio_number[i]
    name=f"mfcc_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    y,sr=librosa.load(file_names[i],sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    mfcc = mfcc[1:, :]
    fig, ax = plt.subplots()
    img=librosa.display.specshow(mfcc, sr=sr, x_axis='time');
    ax.set(title='mfcc escalado')
    plt.savefig(output_path)
    plt.close()
    print(i)

#%%
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Assuming these variables are defined elsewhere:
# path_f, audio_folder, audio_number, file_names, n_mfcc, n_fft, hop_length, sr

# Set the sample rate
sr = 16000

# Change to the directory where audio files are located
os.chdir(path_f)

# Define the output directory for the MFCC plots
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/mfcc_plot_filter'
os.makedirs(output_file, exist_ok=True)

# Loop through files, processing 20 at a time
for i in range(0, len(file_names2), 50):
    a = audio_folder[i]
    b = audio_number[i]
    name = f"mfcc_{a}_{b}.png"
    output_path = os.path.join(output_file, name).replace("\\", "/")
    
    # Load the audio file
    y, sr = librosa.load(file_names2[i], sr=sr)
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Scale the MFCC features
    mfcc = preprocessing.scale(mfcc, axis=1)
    
    # Remove the first coefficient (often considered less relevant for emotion recognition)
    mfcc = mfcc[1:, :]
    
    # Plot the MFCC
    fig, ax = plt.subplots(figsize=(10, 4))  # Optional: Set figure size for better visibility
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    
    ax.set(title='MFCC Escalado')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # bbox_inches='tight' to avoid cut-off labels, dpi for higher resolution
    plt.close()
    
    print(f"Processed file: {i}")

#%%
from tensorflow.keras.utils import to_categorical
print(np.unique(valence))  # Check what values are in valence
valence = np.array(valence) - 1
valence = to_categorical( valence, 5)
activation = np.array(activation) - 1
activation = to_categorical( activation, 5)
dominance = np.array(dominance) - 1
dominance = to_categorical( dominance, 5)


#%% division para datos filtrados
import numpy as np
from sklearn.model_selection import train_test_split

valence2 = np.array(valence2) - 1
valence2 = to_categorical( valence2, 5)
activation2 = np.array(activation2) - 1
activation2 = to_categorical( activation2, 5)
dominance2 = np.array(dominance2) - 1
dominance2 = to_categorical( dominance2, 5)
#%% division datos filtrados
X2 = np.array(data2)    
Y2 = valence2   # convert list to NumPy array
Z2=activation2
W2=dominance2
orden_var = [1, 2, 3, 4,5]  # todos los posibles valores en orden
i=0
# Now you can safely split
X2_train, X2_val, y2_train, y2_val = train_test_split(
    X2,
    Y2,
    test_size=0.2,
    shuffle=True,
    random_state=42
)
X3_train, X3_val, z2_train, z2_val = train_test_split(
    X2,
    Z2,
    test_size=0.2,
    shuffle=True,
    random_state=42
)
X4_train, X4_val, w2_train, w2_val = train_test_split(
    X2,
    W2,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

print("Train shapes:", X2_train.shape, y2_train.shape)
print("Val   shapes:", X2_val.shape,   y2_val.shape, type(y2_val))
#%%

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Suppose your input data has shape: (num_samples, 20, 128), with 1 channel
num_muestras = 3764   # for illustration
frames = 128
caracteristicas = 20
canales = 1  

# Si estamos tratando con una secuencia de características, podríamos tener 1 canal o tantos como características si las tratamos como imágenes.
#%%
# Construcción del modelo
model = Sequential([
    # Primera capa convolucional
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(caracteristicas, frames, canales)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    # Segunda capa convolucional
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    # Tercera capa convolucional
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    Dropout(0.2),
    
    # Aplanar los datos para pasar a capas densas
    Flatten(),
    
    # Capa densa
    Dense(5, activation='softmax')  # Clasificación para 7 emociones
])
# Compilación del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()


#%% modelo en cgpt


import tensorflow as tf
from tensorflow.keras import layers, models
num_muestras = 3764   # for illustration
frames = 128
caracteristicas = 20
canales = 1  

input_shape = (20, 128, 1) 
model = models.Sequential([
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(caracteristicas, frames, canales)),
    layers.Dropout(0.2),  
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.Dropout(0.2),
    
    # Aplanar los datos
    layers.Flatten(),

    layers.Dense(5, activation="softmax" )
])


#%% Compilación del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()
#%%
import time
start_time=time.time()
history_val_2 = model.fit(
    X2_train, 
    y2_train,
    validation_data=(X2_val, y2_val),
    epochs=14,        
    batch_size=32    )
end_time=time.time()
#%%
tiempo=end_time-start_time
ev=model.evaluate(X2_val,y2_val,verbose=0)
print("Perdida y precisión:",ev)
print("tiempo:",tiempo)

#%% matriz de confusion valencia
predictions = model.predict(X2_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y2_val, axis=1)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



cm = confusion_matrix(true_classes, predicted_classes)

# Visualizar la matriz de confusión
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de confusión de valencia')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/ppt3'
name="confusión_val_fil.png"
output_path = os.path.join(output_file, name).replace("\\", "/")
plt.savefig(output_path)
plt.close()
#%% entrenamiento activacion
import time
start_time=time.time()
history_ac_2 = model.fit(
    X3_train, 
    z2_train,
    validation_data=(X3_val, z2_val),
    epochs=14,        
    batch_size=32    )
end_time=time.time()
tiempo=end_time-start_time
ev=model.evaluate(X3_val,z2_val,verbose=0)
print("Perdida y precisión:",ev)
print("tiempo:",tiempo)
#%% matriz de confusion activacion
predictions = model.predict(X3_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(z2_val, axis=1)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



cm = confusion_matrix(true_classes, predicted_classes)

# Visualizar la matriz de confusión
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de confusión de activación')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/ppt3'
name="confusión_acti_fil.png"
output_path = os.path.join(output_file, name).replace("\\", "/")
ax.set(title='waveplot')
plt.savefig(output_path)
plt.close()
#%% entrenamiento dominancia
start_time=time.time()

history_dom_2 = model.fit(
    X4_train, 
    w2_train,
    validation_data=(X4_val, w2_val),
    epochs=14,        
    batch_size=32   )

end_time=time.time()
tiempo=end_time-start_time
ev=model.evaluate(X4_val,w2_val,verbose=0)
print("Perdida y precisión:",ev)
print("tiempo:",tiempo)


#%% matriz de confusion dominancia
predictions = model.predict(X4_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(w2_val, axis=1)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



cm = confusion_matrix(true_classes, predicted_classes)

# Visualizar la matriz de confusión
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de confusión de dominancia')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
output_file = 'C:/Users/ernes/OneDrive/Escritorio/deep_learning/trabajo/ppt3'
name="confusión_domi_fil.png"
output_path = os.path.join(output_file, name).replace("\\", "/")
ax.set(title='waveplot')
plt.savefig(output_path)
plt.close()


#%% extra_input
import numpy as np
Y= np.column_stack((zm, cm, rmsm))

#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the image input
image_input = Input(shape=(20, 128, 1), name="image_input")

# Build the convolutional branch of the CNN
cnn_branch = Conv2D(32, kernel_size=(3, 3), activation="relu")(image_input)
cnn_branch = MaxPooling2D(pool_size=(2, 2))(cnn_branch)
cnn_branch = Conv2D(64, kernel_size=(3, 3), activation="relu")(cnn_branch)
cnn_branch = MaxPooling2D(pool_size=(2, 2))(cnn_branch)
cnn_branch = Flatten()(cnn_branch)

# Define the additional input (extra variables)
extra_input = Input(shape=(3,), name="extra_input")  # Adjust the shape as needed for your extra variables

# Combine the CNN output and the extra input
combined = Concatenate()([cnn_branch, extra_input])
combined = Dense(64, activation="relu")(combined)
combined = Dropout(0.3)(combined)
combined = Dense(10, activation="relu")(combined)

# Final output layer (e.g., for binary classification)
output = Dense(5, activation="softmax")(combined)

# Build the final model
model = Model(inputs=[image_input, extra_input], outputs=output)

#%%
# Compile the model
model.compile(
    loss="categorical_crossentropy ",
    optimizer=Adam(),
    metrics=["accuracy"]
)

# Print the model summary
model.summary()
#%% modelo nuevo valencia

import time
start_time=time.time()
history_val_n = model.fit(
    X2_train, 
    y2_train,
    validation_data=(X2_val, y2_val),
    epochs=14,        
    batch_size=32    )
end_time=time.time()
#%%
tiempo=end_time-start_time
ev=model.evaluate(X2_val,y2_val,verbose=0)
print("Perdida y precisión:",ev)
print("tiempo:",tiempo)
#%%
# Example data (for illustration purposes, adjust to your actual data)
# import numpy as np
# X_train_images = np.random.rand(60000, 28, 28, 1)
# X_train_extra = np.random.rand(60000, 5)
# y_train = np.random.randint(0, 2, size=(60000,))

#Train the model
history = model.fit(
     x={"image_input": data2, "extra_input": Y},
     y=valence2,
     epochs=10,
     batch_size=32,
     validation_split=0.2  # For example, use 20% of the data for validation
 )

# Evaluate the model with test data (adjust X_test_images, X_test_extra, and y_test to your data)
# model.evaluate(
#     x={"image_input": X_test_images, "extra_input": X_test_extra},
#     y=y_test
# )

# Make predictions
# preds = model.predict(
#     {"image_input": X_test_images, "extra_input": X_test_extra}
# )
