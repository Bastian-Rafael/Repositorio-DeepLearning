# En el R
#install.packages("installr")
#library(installr)
#updateR()

install.packages("Rcpp")
install.packages("jsonlite")
install.packages("curl")

devtools::install_github("rstudio/reticulate")
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

devtools::install_github("andrie/deepviz")

library(keras)
library(tensorflow)
library(reticulate)

# En anaconda instalar ambiente y poner tensorflow

install_tensorflow()
install_keras()

################################################################################

library(keras)
library(tensorflow)
library(reticulate)

tf$constant("Hello World!")

mnist <- dataset_mnist()
x_train <- mnist$train$x; x_test <- mnist$test$x
y_train <- mnist$train$y; y_test <- mnist$test$y

par(mfcol=c(4,4))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (i in 1:16) {
  im <- x_train[i,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray(1-(0:255)/255),
        xaxt='n', main=paste(y_train[i]))
}

# Transformación de los arreglos (reshape)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# Escalamiento de los datos (rescale)
x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')


summary(model)


library(deepviz)
library(magrittr)
model %>% plot_model()


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% fit(
  x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)


plot(history)


model %>% evaluate(x_test, y_test)


predicciones = model %>%
  predict(x_test) %>%
  k_argmax() %>%
  array_reshape(., dim = c(10000, 1)) %>%
  as.vector()


predicciones[1:5]


x_test2 <- mnist$test$x
par(mfrow=c(1,5))
par(mar=c(0, 0, 0, 0), xaxs='i', yaxs='i')
for (i in 1:5) {
  im <- x_test2[i,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray(1-(0:255)/255),
        xaxt='n')
}


table(Estimado = predicciones,
      Observado = as.vector(k_argmax(y_test)))

################################################################################

library(keras)

mnist <- dataset_mnist()
mnist$train$x[1,,]; mnist$train$y[1] # representacion de un 5
mnist$train$x[2,,]; mnist$train$y[2] # representación de un 0

x_train <- mnist$train$x; y_train <- mnist$train$y

image(t(apply(mnist$train$x[1,,],2,rev)),col=gray(1-(0:255)/255))
image(1:28,1:28,t(apply(x_train[1,,],2,rev)),col=gray(1-(0:255)/255))

par(mfcol=c(4,4))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (i in 1:16) {
  im <- x_train[i,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray(1-(0:255)/255),
        xaxt='n', main=paste(y_train[i]))
}


################################################################################

mnist=keras::dataset_mnist()
x_train=mnist$train$x; y_train=mnist$train$y; rm(mnist)

par(mfcol=c(2,5)); par(mar=c(0,0,3,0),xaxs='i',yaxs='i')
for(i in 0:9){
  a=which(y_train==i)[sample(1:table(y_train)[i+1],1)]
  image(t(apply(x_train[a,,],2,rev)),col=gray(1-(0:255)/255),
        axes=F,main=paste(y_train[a],"\n Obs:",a)); rm(a,i)
}



