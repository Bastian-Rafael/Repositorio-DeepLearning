summary(application_data)
attach(application_data)
application_data2=na.omit(application_data)
cor(na.omit(cbind(TARGET,AMT_CREDIT,AMT_INCOME_TOTAL,AMT_GOODS_PRICE,HOUR_APPR_PROCESS_START)))

cor(TARGET,application_data[,c(-1,-2)])

mlog=glm(TARGET~DAYS_EMPLOYED+REGION_RATING_CLIENT_W_CITY+EXT_SOURCE_3+AMT_CREDIT+NAME_CONTRACT_TYPE+NAME_INCOME_TYPE+NAME_INCOME_TYPE+AMT_INCOME_TOTAL+AMT_GOODS_PRICE+HOUR_APPR_PROCESS_START,
    family ="binomial",data=application_data2)

summary(mlog)
pred=predict(mlog)
library(pROC)
attach(application_data2)
roc_fumar=roc(TARGET,mlog$fitted.values, auc=TRUE)
plot.roc(roc_fumar,print.thres="best",print.auc=TRUE,main='Curva ROC')
library(tidyverse)
ap=application_data %>% 
  select_if(is.numeric)
summary(ap)

ap=ap[,-1]
corap=cor(ap)
corap[,1]
order(abs(corap[,1]))
abs(corap[c(9,21,81),1])
corap[21,1]




table(transaction_dataset$FLAG)

table(transaccion2$FLAG)
table(transaccion$FLAG)
library(dplyr)
transaccion=transaction_dataset %>% 
  select_if(is.numeric)
summary(transaccion2)

cor(transaccion)[,3]
num=order(abs(cor(transaccion2)[,3]))
cor_transacciones=cor(transaccion2)[,3]
cor_transacciones[num]
attach(transaccion2)
Mcrip=glm(FLAG~transaccion2$`Avg min between received tnx`+transaccion2$`Time Diff between first and last (Mins)`
    +transaccion2$`Sent tnx`+transaccion2$`Received Tnx`+transaccion2$`total transactions (including tnx to create contract`
    ,family = 'binomial')


Mcrip2=glm(FLAG~transaccion2$`Avg min between received tnx`+transaccion2$`Time Diff between first and last (Mins)`
          +transaccion2$`Sent tnx`+transaccion2$`Received Tnx`+transaccion2$`total transactions (including tnx to create contract`
          +transaccion2$`Unique Sent To Addresses`,family = 'binomial')


Mcrip3=glm(FLAG~transaccion2$`Avg min between received tnx`+transaccion2$`Time Diff between first and last (Mins)`
           +transaccion2$`Sent tnx`+transaccion2$`Received Tnx`+transaccion2$`total transactions (including tnx to create contract`
           +transaccion2$`Unique Sent To Addresses`+transaccion2$`Total ERC20 tnxs`,family = 'binomial')


Mcripfinal=glm(transaccion$FLAG~transaccion$`Avg min between received tnx`+transaccion$`Time Diff between first and last (Mins)`
          +transaccion$`Sent tnx`+transaccion$`Received Tnx`+transaccion$`total transactions (including tnx to create contract`+
            transaccion$`avg val sent`,family = 'binomial')

########Analisis exploratorio de los datos################
attach(transaccion)
Base_final=data.frame(FLAG,`Avg min between received tnx`,`Time Diff between first and last (Mins)`,`Sent tnx`,
                      `Received Tnx`,`total transactions (including tnx to create contract`,`avg val sent`)


summary(Base_final)





roccf=roc(transaccion$FLAG,Mcripfinal$fitted.values)
plot.roc(roccf,print.thres='best',print.auc=T)


num1=order(abs(cor(transaccion)[,3]))
cor_transacciones2=cor(transaccion)[,3]
cor_transacciones2[num1]




summary(Mcripfinal)

summary(Mcrip2)
base_fraude_of=data.frame(FLAG,transaccion2$`Avg min between received tnx`,transaccion2$`Time Diff between first and last (Mins)`
,transaccion2$`Sent tnx`,transaccion2$`Received Tnx`,transaccion2$`total transactions (including tnx to create contract`)
library(PerformanceAnalytics)
chart.Correlation(base_fraude_of)
library(pROC)

rocc=roc(transaccion2$FLAG,Mcrip$fitted.values)
plot.roc(rocc,print.thres='best',print.auc=T)

rocc2=roc(transaccion2$FLAG,Mcrip2$fitted.values)
plot.roc(rocc2,print.thres='best',print.auc=T)


rocc3=roc(transaccion2$FLAG,Mcrip3$fitted.values)
plot.roc(rocc3,print.thres='best',print.auc=T)

transaccion0=filter(transaccion2,transaccion2$FLAG==0)
transaccion1=filter(transaccion2,transaccion2$FLAG==1)
summary(transaccion1)
summary(transaccion0)
base_est=scale(base_fraude_of,center = T)
chart.Correlation(base_est)
cor(base_est)
hist(base_fraude_of[,1])
hist(base_fraude_of[,2])
hist(base_fraude_of[,3])
hist(base_fraude_of[,4])
hist(base_fraude_of[,5])
hist(base_fraude_of[,6])
########### diagrama de cajas################
boxplot(base_fraude_of[,2],FLAG)
boxplot(base_fraude_of[,3],FLAG)
boxplot(base_fraude_of[,4],FLAG)
boxplot(base_fraude_of[,5],FLAG)
boxplot(base_fraude_of[,6],FLAG)
table(FLAG)
base_fraude_of0=filter(base_fraude_of,base_fraude_of[,1]==0)
base_fraude_of1=filter(base_fraude_of,base_fraude_of[,1]==1)
summary(base_fraude_of0)
summary(base_fraude_of1)
#######preparacion de la base de datos########################

cor(transaccion)[,3]
base=data.frame(transaccion$FLAG,transaccion$`Avg min between received tnx`,transaccion$`Time Diff between first and last (Mins)`,
                transaccion$`Sent tnx`,transaccion$`Received Tnx`,transaccion$`avg val sent`)

###############particion de labase de datos####################
muestra=sample(1:9841,8856)
fraude_train=base[muestra,]
fraude_test=base[-muestra,]

library(openxlsx)

write.xlsx(fraude_train,'fraude_train.xlsx')
write.xlsx(fraude_test,'fraude_test.xlsx')

library(readxl)
fraude_test <- read_excel("fraude_test.xlsx")
fraude_train <- read_excel("fraude_train.xlsx")

############################################3
library(e1071)
e1071::tune(svm,factor(base_fraude[,1])~base_fraude[,2]+base_fraude[,3]+base_fraude[,4]+base_fraude[,5],data=base_fraude,
            kernel = 'radial',ranges=list(cost = c(0.5, 1,2, 5,10)))

fraude_train=as.data.frame(fraude_train)
fraude_test=as.data.frame(fraude_test)

###### modelo logistico##########
tiempo_ini_log= Sys.time()
mod_logistic=glm(fraude_train[,1]~fraude_train[,2]+fraude_train[,3]+fraude_train[,4]+
                   fraude_train[,5]+fraude_train[,6],family=binomial(link='logit'))
tiempo_fin_log= Sys.time()
tiempo_fin_log-tiempo_ini_log
summary(mod_logistic)

library(pROC)

roc_log=roc(fraude_train[,1],mod_logistic$fitted.values)
mod_logistic$fitted.values
plot.roc(roc_log,print.thres = 'best',print.auc = T,main='Curva AUC modelo logistico')
library(caret)
############tabla de confusion#############
pred_train_log=ifelse(mod_logistic$fitted.values>0.337,1,0)
confusionMatrix(as.factor(pred_train_log),as.factor(fraude_train[,1]),positive='1')
################predicciones#############
sigmoide=function(x){
  sigmoid=1/(1+exp(-x))
  return(sigmoid)
}

X=as.matrix(data.frame(rep(1,985),fraude_test[,-1]))
B=mod_logistic$coefficients
XB=X%*%B
probabilidades_log=sigmoide(XB)
dim(probabilidades_log)

roc_log_pred=roc(fraude_test[,1],probabilidades_log)

plot.roc(roc_log_pred,print.thres = 'best',print.auc = T,main='Curva AUC Modelo logistico')


pred_log=ifelse(probabilidades_log>=0.337,1,0)

confusionMatrix(as.factor(pred_log),as.factor(fraude_test[,1]),positive = '1')

########Aplicacion del modelo de redes neuronales
library(keras)
library(tensorflow)
library(reticulate)
#################modelo 2###############
mod_RN2=keras_model_sequential()

mod_RN2%>%layer_dense(units = 5,activation = 'relu',input_shape = c(5))%>%
  layer_dense(units = 128,activation = 'relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=64,activation='relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=32,activation='relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=1,activation='sigmoid')
summary(mod_RN2)
##################Ajuste del modelo###############

callback2<- list(callback_model_checkpoint("C:/Users/alana/OneDrive/Documentos/modelo5.keras"))
mod_RN2%>%compile(loss = 'binary_crossentropy',metrics=c('accuracy'),optimizer=optimizer_adam())
tiempo_ini_RNN= Sys.time() 
###################

###############
historia=mod_RN2%>%fit(as.matrix(fraude_train[,-1]),as.matrix(fraude_train[,1]),epochs=200,batch_size = 100,validation_split=0.10
,callbacks = callback2)
plot(historia)
tiempo_fin_RNN= Sys.time()
tiempo_fin_RNN-tiempo_ini_RNN

mod_RN2%>%evaluate(as.matrix(fraude_test[,-1]),as.matrix(fraude_test[,1]))

prob_2RN = mod_RN3 %>% 
  predict(as.matrix(fraude_test[,-1])) %>% 
  array_reshape(., dim = c(985, 1)) %>% 
  as.vector()

####################################################3
hist((prob_2RN),main='distribucion de las probabilidades RN',xlab = 'recorrido',ylab='frecuencias')



##########evaluacion de predicciones############
library(pROC)
roc_RN2=roc(as.matrix(fraude_test[,1]),prob_2RN)
plot.roc(roc_RN3,print.auc = T,print.thres = 'best',main='Curva ROC modelo RN',xlim=c(1,0))

predic_RN2=ifelse(prob_3RN>=0.283,1,0)
library(caret)
cf_RN2=confusionMatrix(as.factor(predic_RN2),as.factor(as.matrix(fraude_test[,1])),positive='1')
cf_RN2$byClass
#############Modelo 3###############

library(keras)
library(tensorflow)
library(reticulate)

mod_RN3=keras_model_sequential()

mod_RN3%>%layer_dense(units = 5,activation = 'relu',input_shape = c(5))%>%
  layer_dense(units = 50,activation = 'relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=32,activation='relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=20,activation='relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=1,activation='sigmoid')
summary(mod_RN3)

callback3<- list(callback_model_checkpoint("C:/Users/alana/OneDrive/Documentos/modelo6.keras"))
mod_RN3%>%compile(loss = 'binary_crossentropy',metrics=c('accuracy'),optimizer=optimizer_adam())
tiempo_ini_RNN= Sys.time() 
###################

###############
historia3=mod_RN3%>%fit(as.matrix(fraude_train[,-1]),as.matrix(fraude_train[,1]),epochs=200,batch_size = 100,validation_split=0.10
                       ,callbacks = callback2)
plot(historia3)
tiempo_fin_RNN= Sys.time()
tiempo_fin_RNN-tiempo_ini_RNN

mod_RN3%>%evaluate(as.matrix(fraude_test[,-1]),as.matrix(fraude_test[,1]))

prob_3RN = mod_RN3 %>% 
  predict(as.matrix(fraude_test[,-1])) %>% 
  array_reshape(., dim = c(985, 1)) %>% 
  as.vector()

####################################################3
hist((prob_3RN),main='distribucion de las probabilidades RN',xlab = 'recorrido',ylab='frecuencias')



##########evaluacion de predicciones############
library(pROC)
roc_RN3=roc(as.matrix(fraude_test[,1]),prob_3RN)
plot.roc(roc_RN3,print.auc = T,print.thres = 'best',main='Curva ROC modelo RN',xlim=c(1,0))

predic_RN3=ifelse(prob_3RN>=0.110,1,0)
library(caret)
cf_RN3=confusionMatrix(as.factor(predic_RN3),as.factor(as.matrix(fraude_test[,1])),positive='1')
cf_RN3$byClass












#################modelo 1###############
mod_RN=keras_model_sequential()

mod_RN%>%layer_dense(units = 5,activation = 'relu',input_shape = c(5))%>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 20,activation = 'relu')%>%layer_dropout(rate=0.5)%>%
  layer_dense(units=15,activation='relu')%>%layer_dropout(rate=0.3)%>%
  layer_dense(units=10,activation='relu')%>%layer_dropout(rate=0.2)%>%
  layer_dense(units=1,activation='sigmoid')
summary(mod_RN)
##################Ajuste del modelo###############
mod_RN%>%compile(loss = 'binary_crossentropy',metrics=c('accuracy')
                 ,optimizer=optimizer_adam())

mod_RN%>%fit(as.matrix(fraude_train[,-1]),fraude_train[,1],epochs=100,batch_size = 100,validation_split =as.matrix(fraude_test[,-1]) )

mod_RN%>%evaluate(as.matrix(fraude_test[,-1]),fraude_test[,1])

prob_RN = mod_RN %>% 
  predict(as.matrix(fraude_test[,-1])) %>% 
  array_reshape(., dim = c(985, 1)) %>% 
  as.vector()


pred_RN = mod_RN %>% 
  predict(as.matrix(fraude_test[,-1])) %>%
  array_reshape(., dim = c(985, 1)) %>% 
  as.vector() 

  pred_RN
  ##########evaluacion de predicciones############
roc_RN=roc(fraude_test[,1],prob_RN)
plot.roc(roc_RN,print.auc = T,print.thres = 'best',main='Curva AUC modelo RN')

####################Modelo SVM################
library(e1071)
cost_svm=tune(svm,factor(fraude_train[,1])~fraude_train[,2]+fraude_train[,3]+fraude_train[,4]+fraude_train[,5],data=fraude_train,
     kernel = 'radial',ranges=list(cost = c(20),gamma = c(0.20,0.5, 1, 2, 3, 4, 5, 10)))
plot(cost_svm)

tiempo_ini_SVM= Sys.time() 
mod_svm=svm(as.factor(fraude_train[,1])~.,data=fraude_train[,-1],kernel='radial'
            ,cost=20,gamma=10,decision.values=T,probability=T)
tiempo_fin_SVM= Sys.time()

tiempo_SVM=tiempo_ini_SVM-tiempo_fin_SVM

predicciones_svm=predict(mod_svm,newdata = fraude_train[,-1],probability = T)
prob_svm=attr(predicciones_svm,'probabilities')[,2]

roc_svm=roc(fraude_train[,1],prob_svm)
plot.roc(roc_svm,print.auc = T,print.thres = 'best')

mod_svm$probB
mod_svm$probA
support_vectors=mod_svm$SV
coefficients= mod_svm$coefs
rho=mod_svm$rho


rbf_kernel = function(x, y, gamma) {
  exp(-gamma * sum((x - y)^2))
}

mx=mod_svm$x.scale
mx=data.frame(mx)
fraude_test_scaled <- scale(fraude_test[,-1], 
                            center = mx$scaled.center, 
                            scale = mx$scaled.scale)


decision_values <- apply(fraude_test_scaled, 1, function(test_point) {
  kernel_values <- apply(support_vectors, 1, function(sv) rbf_kernel(test_point, sv, mod_svm$gamma))
  sum(coefficients * kernel_values) - rho
})

decision_values

decision_values
a=mod_svm$probA
b=mod_svm$probB
dec_train=mod_svm$decision.values
a*decision_values+b

prob_svm_pred=sigmoide((a*decision_values+b))

roc_svm_pred=roc(fraude_test[,1],prob_svm_pred)
plot.roc(roc_svm_pred,print.auc = T,print.thres = 'best',main='Curva ROC SVM')

pred_svm=ifelse(prob_svm_pred>=0.105,1,0)

confusionMatrix(as.factor(pred_svm),as.factor(fraude_test[,1]),positive='1')
####################modelos##############

cf_RN$byClass
cf_
cf_RN2$byClass
cf_RN3$byClass
