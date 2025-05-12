rm(list=ls()); shell("cls"); graphics.off() ###

#data=readxl::read_excel(file.choose())
data=readxl::read_excel("C:/Users/nesto/Desktop/Musikita.xlsx")

data$Continente=ifelse(data$País %in% c("Belgium","Poland","Switzerland",
                                  "Portugal","France","UK","Norway","Sweden","Ireland",
                                  "Austria","Germany","Netherlands","Denmark","Spain",
                                  "Italy","Finland"),"Europa",
                 
                 ifelse(data$País %in% c("Costa Rica","Colombia","Mexico","Peru",
                                         "Brazil","Ecuador","Chile","Argentina","Canada","USA"),"América",
                        
                 ifelse(data$País %in% c("Philippines","Singapore","Malaysia",
                                                "Taiwan","Indonesia","Turkey"),"Asia",
                               
                               ifelse(data$País %in% c("New Zealand","Australia"),"Oceanía",NA))))

data1=data=data[data$País!="Global",]; data=data[,-c(4,5,20,22)]

data=fastDummies::dummy_cols(data,select_columns=c("País","Formato","Género","Continente"),
                             remove_selected_columns=T)

est=c("Popularidad","Top_Max","Bailable","Energía","Tono","Volumen","Hablado","Acústico",
      "Instrumental","Valencia","BPM","Duración_ms","Días_Lanzamiento")
data[est]=as.data.frame(lapply(data[est],scales::rescale,to=c(0,1))); rm(est)

set.seed(123)
index=sample(1:nrow(data),size=.8*nrow(data))
X_train=as.matrix(data[index,-1]); y_train=as.matrix(data[index,1])
X_test=as.matrix(data[-index,-1]); y_test=as.matrix(data[-index,1]); rm(index)

library(keras)
library(tensorflow)
library(dplyr)

####################################################################################

model=keras_model_sequential(name="Modelo_prueba") %>%
  layer_dense(units=64,activation="relu", 
              kernel_regularizer=regularizer_l2(1e-6), 
              input_shape=ncol(X_train)) %>%
              layer_dropout(rate=.5) %>%
  layer_dense(units=32,activation="relu", 
              kernel_regularizer=regularizer_l2(1e-6)) %>%
              layer_dropout(rate=.5) %>%
  layer_dense(units=16,activation="relu", 
              kernel_regularizer=regularizer_l2(1e-6)) %>%
              layer_dropout(rate=.5) %>%
  layer_dense(units=1,activation="linear"); summary(model)

model %>% compile(optimizer=optimizer_adam(learning_rate=1e-4),loss="mse",metrics=c("mae","mse"))

early_stopping=callback_early_stopping(monitor="val_loss",patience=10)
reduce_lr=callback_reduce_lr_on_plateau(monitor="val_loss", 
                                        factor=.2, 
                                        patience=5, 
                                        min_lr=1e-6)
t=system.time({
set.seed(1234)
history=model %>% fit(X_train,y_train,epochs=100,batch_size=32,
                  validation_split=.2,callbacks=list(early_stopping,reduce_lr))
}); plot(history); t[3]/60

evaluation=model %>% evaluate(X_test,y_test); evaluation

y_pred=model %>% predict(X_test)

plot(y_test,y_pred,xlab="Valor Real",ylab="Predicción",main="Predicciones vs. Valores Reales")
abline(0,1,col="red")





descale=function(x,min,max){
  return(x*(max-min)+min)}

Y_test=as.data.frame(descale(y_test,min(data1$Popularidad),max(data1$Popularidad)))
Y_pred=as.data.frame(descale(y_pred,min(data1$Popularidad),max(data1$Popularidad)))

a=cbind(Real=Y_test,Prediccion=Y_pred); a[,3]=abs(a[,1]-a[,2])
names(a)=c("Real","Prediccion","Abs"); a

median(data1$Popularidad)
mean(data1$Popularidad)


# 7.589667 min 
# loss         mae         mse 
# 0.001548853 0.017651396 0.001482622 
