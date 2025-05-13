### Preparar datos ###
# Primero se filtran irregularidades #
base <- Respuestas.Casen %>% filter((CIUO_N1>=0 & CIUO_N1<=9) | (CIUO_N2>=0 & CIUO_N2<=96) | 
                                      (CIUO_N3>=0 & CIUO_N3<=962) | (CIUO_N4>=0 & CIUO_N4<=9629) )
# Se eliminan las observaciones de la categoria 0 en el nivel 1 #
base <- base %>% filter(CIUO_N1!=0)

num=c(1,2,3,4,5,6,7,8,9)
set.seed(999)
muestra=dplyr::sample_n(base[base$CIUO_N1 %in% num,],80000)
table(muestra$CIUO_N2)
muestra <- base
########################################################################
### Separar la base en datos de entrenamiento y de prueba ###
set.seed(6173)
train_indices <- sample(seq_len(nrow(muestra)), size = 0.7 * nrow(muestra))
train_data <- muestra[train_indices, ]
test_data <- muestra[-train_indices, ]

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
table(base$CIUO_N2)
mapeo <- c(`11`=0,`12`=1,`13`=2,`14`=3,`21`=4,`22`=5,`23`=6,`24`=7,`25`=8,`26`=9,`31`=10,
           `32`=11,`33`=12,`34`=13,`35`=14,`36`=15,`41`=16,`42`=17,`43`=18,`44`=19,`51`=20,`52`=21,`53`=22,
           `54`=23,`61`=24,`62`=25,`63`=26,`71`=27,`72`=28,`73`=29,`74`=30,`75`=31,`81`=32,`82`=33,`83`=34,
           `91`=35,`92`=36,`93`=37,`94`=38,`95`=39,`96`=40)

# Reindexar las etiquetas en el conjunto de entrenamiento y prueba
train_labels <- dplyr::recode(train_data$CIUO_N2, !!!mapeo)
test_labels <- dplyr::recode(test_data$CIUO_N2, !!!mapeo)

train_labels <- as.array(train_labels)  # Etiquetas para entrenamiento
test_labels <- as.array(test_labels)    # Etiquetas para prueba

train_padded <- as.matrix(train_padded)
test_padded <- as.matrix(test_padded)

# Definir el tamaño del batch y del buffer
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


### Ajuste del modelo ###
model_lvl2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, mask_zero = TRUE) %>%
  bidirectional(layer_lstm(units = 128, return_sequences = TRUE)) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dense(units = 64, activation = 'relu', kernel_regularizer = regularizer_l1_l2(0.01)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = 'relu', kernel_regularizer = regularizer_l1_l2(0.01)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Resumen del modelo
summary(model)

early_stop <- callback_early_stopping(
  monitor = "val_loss",       # Métrica a monitorear, normalmente 'val_loss' o 'val_accuracy'
  patience = 3,               # Número de épocas sin mejora antes de detener el entrenamiento
  restore_best_weights = TRUE # Restaurar los pesos del modelo que alcanzaron el mejor resultado
)


model_lvl2 %>% compile(
  loss = 'sparse_categorical_crossentropy',  
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Entrenar el modelo
tiempo_ini_lvl2 = Sys.time()
history_lvl2 <- model_lvl2 %>% fit(
  train_dataset,      # Conjunto de datos de entrenamiento
  epochs = 10,         # Número de épocas
  validation_data = test_dataset,  # Conjunto de datos de validación
  callbacks = list(early_stop)
)
tiempo_fin_lvl2 = Sys.time()

tiempo_ini_lvl2
tiempo_fin_lvl2
# Hacer predicciones sobre los datos de prueba
predicciones_lvl2 <- model_lvl2 %>% predict(test_dataset)

# Convertir las predicciones en etiquetas (la clase con la mayor probabilidad)
etiquetas_predichas_lvl2 <- apply(predicciones_lvl2, 1, which.max) - 1

# Comparar con las etiquetas reales
# Suponiendo que y_test contiene las etiquetas reales
matriz_confusion_lvl2 <- table(Predicted = etiquetas_predichas_lvl2, Actual = test_labels)
print(matriz_confusion_lvl2)
confusion_lvl2<-as.data.frame(matriz_confusion_lvl2)
# Calcular la precisión
precision_lvl2 <- sum(etiquetas_predichas_lvl2 == test_labels) / length(test_labels)
print(paste("Precisión: ", precision_lvl2))
# Guardar un dataframe como archivo CSV
write.csv(confusion_lvl2, file = "confusion_lvl2.csv", row.names = FALSE)

# Guardar el modelo completo en un archivo .h5
save_model_hdf5(model_lvl2, filepath = "red_lvl2.h5")
# Cargar el modelo desde un archivo .h5
model <- load_model_hdf5("mi_modelo.h5")