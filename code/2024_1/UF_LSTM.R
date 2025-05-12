# Cargar las librerías necesarias
library(keras)
library(tidyverse)
library(ggplot2)
library(readxl)

setwd("C:/Users/naaat/OneDrive/Escritorio/MEJOR MODELO")



serie_UF2=serie_UF
serie_UF2$Periodo=as.Date(serie_UF2$Periodo)
serie_dolar2=serie_dolar
serie_dolar2$Periodo=as.Date(serie_dolar2$Periodo)

serie_final=merge(x=serie_dolar2,y=serie_UF2[,-3],by="Periodo") #Inner Join
serie_final=na.omit(serie_final)
data = serie_final %>% 
  rename("Dolar" = "1.Dólar observado",
         "UF" = "1.Unidad de fomento (UF)") %>% 
  # dplyr::filter(Periodo > as.Date("2019-12-31")) %>% 
  select(-Periodo)

# Escalar la variable Dolar
data_scaled <- data %>%
  mutate(Dolar = scale(Dolar))

# Crear una función para generar las secuencias de los rezagos de "Dolar"
create_sequences <- function(data, n_lags) {
  X <- NULL
  y <- NULL
  for (i in (n_lags + 1):nrow(data)) {
    dolar_lags <- data[(i - n_lags):(i - 1), "Dolar"]
    X <- rbind(X, dolar_lags)
    y <- c(y, data[i, "Dolar"])
  }
  list(X = array(X, dim = c(nrow(X), n_lags, 1)), y = y)
}

# Definir el número de lags
n_lags <- 60
# Dividir los datos en conjunto de entrenamiento y prueba
set.seed(123)  # Para reproducibilidad
train_size <- round(0.8 * length(y))
mse <- matrix(0,nrow=length(n_lags),ncol=2) 
for(i in 1:length(n_lags))
{
  # Generar las secuencias
  sequences <- create_sequences(as.matrix(data_scaled), n_lags[i])
  X <- sequences$X
  y <- sequences$y
  X_train <- X[1:train_size,,]
  y_train <- y[1:train_size]
  X_test <- X[(train_size+1):length(y),,]
  y_test <- y[(train_size+1):length(y)]

  # Crear el modelo LSTM
  model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(n_lags[i], 1)) %>%
  layer_dense(units = 1)

  # Compilar el modelo
  model %>% compile(
  loss = 'mse',
  optimizer = 'adam'
  )

  # Definir los callbacks
  checkpoint <- callback_model_checkpoint(filepath = "mejor_modelo.h5", 
                                        save_best_only = TRUE, 
                                        monitor = "val_loss", 
                                        mode = "min")

  early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
  reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)

  # Entrenar el modelo
  history <- model %>% fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = list(checkpoint, early_stopping, reduce_lr),
    verbose = 1
  )

  system.time(history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(checkpoint, early_stopping, reduce_lr),
  verbose = 1
  ))

  # Cargar el mejor modelo guardado
  best_model <- load_model_hdf5("mejor_modelo.h5")

  # Evaluar el mejor modelo
  mse[i,1]= best_model %>% evaluate(X_test, y_test)
  mse[i,2]= best_model %>% evaluate(X_train, y_train)

  library(Metrics)
  mse_value <- mse(y_test, predictions_test); mse_value
  
  
  # Hacer predicciones con el mejor modelo
  predictions_train <- best_model %>% predict(X_train)
  predictions_test <- best_model %>% predict(X_test)
}

plot(n_lags,mse[,1],pch=20,lwd=2,ylim=c(0,0.0025),ylab="MSE",xlab="N Lags")
lines(n_lags,mse[,1])
points(n_lags,mse[,2],pch=20,lwd=2,col="blue")
lines(n_lags,mse[,2],col="blue")
legend(x = "topright",          # Position
       legend = c("Train", "Test"),  # Legend texts
       lty = c(1, 1),           # Line types
       col = c("blue", "black"),           # Line colors
       pch = 20,cex=0.8)

n_lags[which.min(mse[,1])]
n_lags[which.min(mse[,2])]


###MODELO FINAL
n_lags=60
# Generar las secuencias
sequences <- create_sequences(as.matrix(data_scaled), n_lags)
X <- sequences$X
y <- sequences$y
X_train <- X[1:train_size,,]
y_train <- y[1:train_size]
X_test <- X[(train_size+1):length(y),,]
y_test <- y[(train_size+1):length(y)]

# Crear el modelo LSTM
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(n_lags, 1)) %>%
  layer_dense(units = 1)

# Compilar el modelo
model %>% compile(
  loss = 'mse',
  optimizer = 'adam'
)

# Definir los callbacks
checkpoint <- callback_model_checkpoint(filepath = "mejor_modelo.h5", 
                                        save_best_only = TRUE, 
                                        monitor = "val_loss", 
                                        mode = "min")

early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)

# Entrenar el modelo
history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(checkpoint, early_stopping, reduce_lr),
  verbose = 1
)

system.time(history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(checkpoint, early_stopping, reduce_lr),
  verbose = 1
))

# Cargar el mejor modelo guardado
best_model <- load_model_hdf5("mejor_modelo.h5")

# Evaluar el mejor modelo
best_model %>% evaluate(X_test, y_test)
best_model %>% evaluate(X_train, y_train)

# Hacer predicciones con el mejor modelo
predictions_train <- best_model %>% predict(X_train)
predictions_test <- best_model %>% predict(X_test)


  # Crear un dataframe para el gráfico
df <- data.frame(
  Index = c(1:length(y_train), (length(y_train) + 1):(length(y_train) + length(y_test))),
  Real = c(y_train, y_test),
  Predicción = c(predictions_train, predictions_test),
  Conjunto = rep(c("Entrenamiento", "Prueba"), c(length(y_train), length(y_test)))
)

# Graficar resultados con colores diferentes para lo observado y lo predicho, separado por conjunto
ggplot(df, aes(x = Index)) +
  geom_line(aes(y = Real, color = Conjunto, linetype = "Observado")) +
  #geom_line(aes(y = Predicción, color = Conjunto, linetype = "Predicho")) +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Prueba" = "green")) +
  scale_linetype_manual(values = c("Observado" = "solid", "Predicho" = "dashed")) +
  labs(title = "Resultados del Mejor Modelo LSTM: Observado vs Predicho",
       x = "Índice de Muestra", y = "Valor de Dolar (escalado)") +
  theme_minimal() +
  theme(legend.title = element_blank())

ggplot(df, aes(x = Index)) +
  
  #geom_line(aes(y = Real, color = Conjunto, linetype = "Observado")) +
  geom_line(aes(y = Predicción, color = Conjunto, linetype = "Predicho")) +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Prueba" = "green")) +
  scale_linetype_manual(values = c("Observado" = "solid", "Predicho" = "dashed")) +
  labs(title = "Resultados del Mejor Modelo LSTM: Observado vs Predicho",
       x = "Índice de Muestra", y = "Valor de Dolar (escalado)") +
  theme_minimal() +
  theme(legend.title = element_blank())

