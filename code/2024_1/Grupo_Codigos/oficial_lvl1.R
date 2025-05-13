library(tensorflow)
library(tidyverse)
library(text) 
library(textTinyR)  
library(keras)
library(tm)
library(tfdatasets)

### Preparar datos ###
# Primero se filtran irregularidades #
base <- Respuestas.Casen %>% filter((CIUO_N1>=0 & CIUO_N1<=9) | (CIUO_N2>=0 & CIUO_N2<=96) | 
                                      (CIUO_N3>=0 & CIUO_N3<=962) | (CIUO_N4>=0 & CIUO_N4<=9629) )
# Se eliminan las observaciones de la categoria 0 en el nivel 1 #
base <- base %>% filter(CIUO_N1!=0)

muestra <- base

########################################################################
### Separar la base en datos de entrenamiento y de prueba ###
set.seed(6173)
train_indices <- sample(seq_len(nrow(muestra)), size = 0.7 * nrow(muestra))
train_data <- muestra[train_indices, ]
test_data <- muestra[-train_indices, ]

########################################################################
########## Procesamiento de lenguaje natural #########
### Extraer el texto de las bases a trabajar ###
train_data$text=paste(train_data$Ocupación,train_data$Tarea)
text_data=train_data$text # extraccion texto entrenamiento #
test_data$text=paste(test_data$Ocupación,test_data$Tarea)
text_data_Test=test_data$text # extraccion texto prueba #

### Se realiza una limpieza de los textos extraidos (stopwords) ###
remove_stopwords <- function(text_data) {
  # Crear un corpus a partir del texto
  corpus <- Corpus(VectorSource(text_data))
  
  # Definir la lista de stopwords en español
  stopwords_es <- stopwords("spanish")
  
  # Función para eliminar stopwords
  clean_corpus <- tm_map(corpus, content_transformer(function(x) {
    # Convertir a minúsculas
    x <- tolower(x)
    # Eliminar puntuación
    x <- removePunctuation(x)
    # Eliminar números
    x <- removeNumbers(x)
    # Eliminar stopwords
    x <- removeWords(x, stopwords_es)
    # Eliminar espacios en blanco adicionales
    x <- stripWhitespace(x)
    return(x)
  }))
  
  # Convertir el corpus limpio de nuevo a texto
  cleaned_text <- sapply(clean_corpus, as.character)
  
  return(cleaned_text)
}

### Limpiar los textos eliminando stopwords ###
clean_text_data <- remove_stopwords(text_data)
clean_text_data_T <- remove_stopwords(text_data_Test)

### Crear un tokenizador con el texto limpio ###
tokenizer <- text_tokenizer() %>%
  fit_text_tokenizer(clean_text_data)

### vocabulario del tokenizador ###
vocab_size <- length(tokenizer$word_index) + 1  # +1 para incluir el índice 0 (máscara)

### Convertir los textos limpios en secuencias de enteros ###
train_sequences <- texts_to_sequences(tokenizer, clean_text_data)
test_sequences <- texts_to_sequences(tokenizer, clean_text_data_T)

### Pad de las secuencias para que tengan la misma longitud ###
maxlen <- max(sapply(train_sequences, length))  # Puedes ajustar esto según tus necesidades
train_padded <- pad_sequences(train_sequences, maxlen = maxlen)
test_padded <- pad_sequences(test_sequences, maxlen = maxlen)
# Convertir a matrices y etiquetas (suponiendo que tienes etiquetas para entrenamiento y prueba)
# Crear el mapeo con recode
mapeo <- c(`1`=0,`2`=1,`3`=2,`4`=3,`5`=4,`6`=5,`7`=6,`8`=7,`9`=8)


# Reindexar las etiquetas en el conjunto de entrenamiento y prueba
train_labels <- dplyr::recode(train_data$CIUO_N1, !!!mapeo)
test_labels <- dplyr::recode(test_data$CIUO_N1, !!!mapeo)

train_labels <- as.array(train_labels)  # Etiquetas para entrenamiento
test_labels <- as.array(test_labels)    # Etiquetas para prueba

train_padded <- as.matrix(train_padded)
test_padded <- as.matrix(test_padded)

### Definir el tamaño del batch y del buffer ###
batch_size <- 64  # Puedes ajustar esto según la memoria disponible y tus necesidades
buffer_size <- 123967  # Para barajar completamente el conjunto de entrenamiento

# Crear dataset para entrenamiento
train_dataset <- tensor_slices_dataset(list(train_padded, train_labels)) %>%
  dataset_batch(batch_size) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_prefetch(buffer_size)

# Crear dataset para prueba
test_dataset <- tensor_slices_dataset(list(test_padded, test_labels)) %>%
  dataset_batch(batch_size) %>%
  dataset_prefetch(buffer_size)
dim(train_padded)

# Ajustar el tamaño de la última capa y la pérdida
num_classes <- length(unique(train_labels))  # Número de clases
###############################################################################


### Ajuste del modelo ###
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, mask_zero = TRUE) %>%
  bidirectional(layer_lstm(units = 128, return_sequences = TRUE)) %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dense(units = 16, activation = 'relu', kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>%
  layer_dropout(rate = 0.45) %>%
  layer_dense(units = num_classes, activation = 'softmax')

### Parada temprana ###
early_stop <- callback_early_stopping(
  monitor = "val_loss",       
  patience = 3,               
  restore_best_weights = TRUE 
)
### Compilación del modelo ###
model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001, beta_1 = 0.9),
  metrics = c('accuracy')
)

### Entrenar el modelo ###
tiempo_ini = Sys.time()
history <- model %>% fit(
  train_dataset,      # Conjunto de datos de entrenamiento
  epochs = 10,         # Número de épocas
  validation_data = test_dataset,  # Conjunto de datos de validación
  callbacks = list(early_stop)
)
tiempo_fin = Sys.time()

########################### REDULTADOS LVL 1 ##################################
tiempo_ini
tiempo_fin
# Hacer predicciones sobre los datos de prueba
predicciones <- model %>% predict(test_dataset)

# Convertir las predicciones en etiquetas (la clase con la mayor probabilidad)
etiquetas_predichas <- apply(predicciones, 1, which.max) - 1

# Comparar con las etiquetas reales
# Suponiendo que y_test contiene las etiquetas reales
matriz_confusion <- table(Predicted = etiquetas_predichas, Actual = test_labels)
print(matriz_confusion)

# Calcular la precisión
precision <- sum(etiquetas_predichas == test_labels) / length(test_labels)
print(paste("Precisión: ", precision))


### Precición por categoria ###
# precición categoria 1 (0 en la matriz)
round((matriz_confusion[1,1]*100)/sum(matriz_confusion[,1]),3)

# precición categoria 2 (1 en la matriz)
round((matriz_confusion[2,2]*100)/sum(matriz_confusion[,2]),3)

# precición categoria 3 (2 en la matriz)
round((matriz_confusion[3,3]*100)/sum(matriz_confusion[,3]),3)

# precición categoria 4 (3 en la matriz)
round((matriz_confusion[4,4]*100)/sum(matriz_confusion[,4]),3)

# precición categoria 5 (4 en la matriz)
round((matriz_confusion[5,5]*100)/sum(matriz_confusion[,5]),3)

# precición categoria 6 (5 en la matriz)
round((matriz_confusion[6,6]*100)/sum(matriz_confusion[,6]),3)

# precición categoria 7 (6 en la matriz)
round((matriz_confusion[7,7]*100)/sum(matriz_confusion[,7]),3)

# precición categoria 8 (7 en la matriz)
round((matriz_confusion[8,8]*100)/sum(matriz_confusion[,8]),3)

# precición categoria 9 (8 en la matriz)
round((matriz_confusion[9,9]*100)/sum(matriz_confusion[,9]),3)



