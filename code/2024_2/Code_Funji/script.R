rm(list = ls())

# Librerías ####

library(keras)
library(tensorflow)
library(imager)
library(tidyverse)
library(fs)
library(caret)
library(andrews)

# Balanceo de Categorías ####

# Definir ruta principal
data_dir <- "C:/Users/NASa/OneDrive - usach.cl/INGESTA24mk.II/deep learning/k jumpers"
output_dir <- file.path(data_dir, "defungi - Copy")
dir.create(output_dir, showWarnings = FALSE)

# Función para aplicar data augmentation
augment_images <- function(files, target_size, output_folder) {
     num_files <- length(files)
     augment_factor <- ceiling(target_size / num_files)
     augmented_files <- c()
     
     for (file in files) {
          img <- tryCatch(load.image(file), error = function(e) NULL)
          if (is.null(img)) next
          
          for (i in seq_len(augment_factor)) {
               if (is.null(img)) next
               
               # Guardar imagen con sufijo "augmentation"
               original_name <- tools::file_path_sans_ext(basename(file))
               augmented_file <- file.path(output_folder, paste0(original_name, "_(", i, ").jpg"))
               tryCatch(save.image(img, augmented_file), error = function(e) NULL)
               augmented_files <- c(augmented_files, augmented_file)
               
               # Salir si alcanzamos el número objetivo
               if (length(augmented_files) + num_files >= target_size) break
          }
          if (length(augmented_files) + num_files >= target_size) break
     }
     
     return(augmented_files)
}

# Asegurar exactamente 1000 imágenes por carpeta
ensure_exactly_1000 <- function(folder) {
     files <- list.files(folder, full.names = TRUE)
     num_files <- length(files)
     
     if (num_files > 1000) {
          # Si hay más de 1000, eliminar las sobrantes
          files_to_keep <- sample(files, 1000)
          files_to_remove <- setdiff(files, files_to_keep)
          file.remove(files_to_remove)
          cat("Exceso eliminado: ", length(files_to_remove), " imágenes removidas de ", basename(folder), "\n")
     } else if (num_files < 1000) {
          # Si hay menos de 1000, generar imágenes adicionales
          extra_files_needed <- 1000 - num_files
          augment_images(files, 1000, folder)
          cat("Imágenes generadas: ", extra_files_needed, " añadidas a ", basename(folder), "\n")
     }
}

# Crear carpetas de salida y procesar categorías
process_categories <- function(folder_list) {
     for (folder in folder_list) {
          class_name <- basename(folder)
          output_folder <- file.path(output_dir, paste0(class_name, "_1000"))
          dir.create(output_folder, showWarnings = FALSE)
          
          # Determinar el método de procesamiento
          method <- if (class_name %in% c("H1", "H2")) "MAS" else "AUGMENT"
          
          # Ajustar imágenes y garantizar 1000
          files <- list.files(folder, full.names = TRUE)
          if (method == "MAS") {
               sampled_files <- sample(files, 1000)
               file.copy(sampled_files, output_folder)
          } else {
               if (length(files) < 1000) {
                    augmented_files <- augment_images(files, 1000, output_folder)
                    file.copy(files, output_folder)
               } else {
                    sampled_files <- sample(files, 1000)
                    file.copy(sampled_files, output_folder)
               }
          }
          
          # Validar que haya exactamente 1000 imágenes
          ensure_exactly_1000(output_folder)
     }
}

# Lista de carpetas de categorías originales
folder_list <- list.dirs(file.path(data_dir, "defungi"), full.names = TRUE, recursive = FALSE)

# Procesar las categorías
# process_categories(folder_list)

# Contar imágenes en las nuevas carpetas
count_images_per_category <- function(directory) {
     class_folders <- list.dirs(directory, full.names = TRUE, recursive = FALSE) # Excluir el directorio principal
     counts <- sapply(class_folders, function(folder) length(list.files(folder, full.names = TRUE)))
     tibble(
          Category = basename(class_folders),
          Count = counts
     )
}

# Categorías sin balancear
file.path(data_dir, "defungi") %>%
     count_images_per_category %>% 
     ggplot(aes(x = Category, y = Count, fill = Category)) +
     geom_bar(stat = "identity", color = "black") +
     labs(title = "Número de Imágenes por Categoría", x = "Categoría", y = "Cantidad de Imágenes") +
     theme_minimal() +
     scale_fill_brewer(palette = "Paired")

# Categorías balanceadas
processed_counts <- count_images_per_category(output_dir)

# Filtrar únicamente categorías válidas
processed_counts <- processed_counts %>%
     filter(Category %in% c("H1_1000", "H2_1000", "H3_1000", "H5_1000", "H6_1000"))

print(processed_counts)

# Categorías balanceadas
processed_counts %>% 
     ggplot(aes(x = Category, y = Count, fill = Category)) +
     geom_bar(stat = "identity", color = "black") +
     labs(title = "Número de Imágenes por Categoría", x = "Categoría", y = "Cantidad de Imágenes") +
     theme_minimal() +
     scale_fill_brewer(palette = "Paired")

# Listar todas las imágenes dentro de la carpeta Hongos_1000
file_name <- list.files(output_dir, recursive = TRUE, full.names = TRUE)

# Seleccionar 12 imágenes aleatorias
set.seed(99) # Asegurar reproducibilidad
sample_image <- sample(file_name, 12)

# Cargar las imágenes seleccionadas
img <- map(sample_image, function(x) {
     tryCatch(load.image(x), error = function(e) NULL) # Manejar errores en caso de que alguna imagen no se cargue
})

# Filtrar imágenes cargadas correctamente
img <- compact(img) # Eliminar entradas NULL si alguna imagen no pudo cargarse

# Graficar las imágenes
if (length(img) > 0) {
     par(mfrow = c(3, 4)) # Crear una grilla de 3 x 4
     par(mar = c(1, 1, 3, 1)) # Ajustar márgenes
     map(img, plot)
} else {
     cat("No se pudieron cargar imágenes para graficar.\n")
}

# Análisis Exploratorio ####

## Análisis de Colores ####

folder_list2 <- c(paste0(output_dir, "/H1_1000/"),
                  paste0(output_dir, "/H2_1000/"),
                  paste0(output_dir, "/H3_1000/"),
                  paste0(output_dir, "/H5_1000/"),
                  paste0(output_dir, "/H6_1000/")
)

# Generar pixeles aleatorios
set.seed(123)
pixel_sample <- data.frame(x = sample(x = 1:500, size = 1000, replace = TRUE),
                           y = sample(x = 1:500, size = 1000, replace = TRUE))

# Función para extraer colores
extract_rgb <- function(image_path, pixels) {
     img <- load.image(image_path)
     n_pixels <- nrow(pixels)
     red <- numeric(n_pixels)
     green <- numeric(n_pixels)
     blue <- numeric(n_pixels)
     
     for (i in 1:n_pixels) {
          x = pixels[i, 1]
          y = pixels[i, 2]
          red[i] <- img[x, y, , 1]
          green[i] <- img[x, y, , 2]
          blue[i] <- img[x, y, , 3]
     }
     return(list(red = red, green = green, blue = blue))
}

### H1 ####
file_name2 <- map(folder_list2[1],
                  function(x) paste0(x, list.files(x))
                  ) %>% 
     unlist()

file_name2 <- file_name2[sample(1:1000,100)]

rgb <- sapply(file_name2, function(file_path) extract_rgb(file_path, pixel_sample))

red <- rgb[1:300 %% 3 == 1]
green <- rgb[1:300 %% 3 == 2]
blue <- rgb[1:300 %% 3 == 0]

andrews(red, type = 3, clr = 5, ylim = c(-15,15), main = "H1")
andrews(green, type = 3, clr = 5, ylim = c(-15,15), main = "H1")
andrews(blue, type = 3, clr = 5, ylim = c(-15,15), main = "H1")

### H2 ####
file_name2 <- map(folder_list2[2],
                  function(x) paste0(x, list.files(x))
) %>% 
     unlist()

file_name2 <- file_name2[sample(1:1000,100)]

rgb <- sapply(file_name2, function(file_path) extract_rgb(file_path, pixel_sample))

red <- rgb[1:300 %% 3 == 1]
green <- rgb[1:300 %% 3 == 2]
blue <- rgb[1:300 %% 3 == 0]

andrews(red, type = 3, clr = 5, ylim = c(-15,15), main = "H2")
andrews(green, type = 3, clr = 5, ylim = c(-15,15), main = "H2")
andrews(blue, type = 3, clr = 5, ylim = c(-15,15), main = "H2")

### H3 ####
file_name2 <- map(folder_list2[3],
                  function(x) paste0(x, list.files(x))
) %>% 
     unlist()

file_name2 <- file_name2[sample(1:1000,100)]

rgb <- sapply(file_name2, function(file_path) extract_rgb(file_path, pixel_sample))

red <- rgb[1:300 %% 3 == 1]
green <- rgb[1:300 %% 3 == 2]
blue <- rgb[1:300 %% 3 == 0]

andrews(red, type = 3, clr = 5, ylim = c(-15,15), main = "H3")
andrews(green, type = 3, clr = 5, ylim = c(-15,15), main = "H3")
andrews(blue, type = 3, clr = 5, ylim = c(-15,15), main = "H3")

### H5 ####
file_name2 <- map(folder_list2[4],
                  function(x) paste0(x, list.files(x))
) %>% 
     unlist()

file_name2 <- file_name2[sample(1:1000,100)]

rgb <- sapply(file_name2, function(file_path) extract_rgb(file_path, pixel_sample))

red <- rgb[1:300 %% 3 == 1]
green <- rgb[1:300 %% 3 == 2]
blue <- rgb[1:300 %% 3 == 0]

andrews(red, type = 3, clr = 5, ylim = c(-15,15), main = "H5")
andrews(green, type = 3, clr = 5, ylim = c(-15,15), main = "H5")
andrews(blue, type = 3, clr = 5, ylim = c(-15,15), main = "H5")

### H6 ####
file_name2 <- map(folder_list2[5],
                  function(x) paste0(x, list.files(x))
) %>% 
     unlist()

file_name2 <- file_name2[sample(1:1000,100)]

rgb <- sapply(file_name2, function(file_path) extract_rgb(file_path, pixel_sample))

red <- rgb[1:300 %% 3 == 1]
green <- rgb[1:300 %% 3 == 2]
blue <- rgb[1:300 %% 3 == 0]

andrews(red, type = 3, clr = 5, ylim = c(-15,15), main = "H6")
andrews(green, type = 3, clr = 5, ylim = c(-15,15), main = "H6")
andrews(blue, type = 3, clr = 5, ylim = c(-15,15), main = "H6")

## Análisis de Dimensiones ####
get_dim <- function(x){
     img <- load.image(x) 
     df_img <- data.frame(height = height(img),
                          width = width(img),
                          filename = x
     )
     return(df_img)
}

file_name2 <- map(folder_list2,
                  function(x) paste0(x, list.files(x))
) %>% 
     unlist()

file_dim <- map_df(file_name2, get_dim)
file_dim %>% 
     ggplot(aes(x=width, y=height)) +
     geom_point(color="orange3", fill="orange1", alpha=0.5)
table(file_dim$height)
table(file_dim$width)

# Modelo B&N ####

# Generadores de datos
train_image_array_gen <- image_data_generator(
     rescale = 1 / 255,
     rotation_range = 30,
     width_shift_range = 0.2,
     height_shift_range = 0.2,
     shear_range = 0.2,
     zoom_range = 0.2,
     horizontal_flip = TRUE,
     fill_mode = "nearest",
     validation_split = 0.2
)

# Cargar datos de entrenamiento
batch_size = 32
train_data_gen <- flow_images_from_directory(
     target_size = c(150L, 150L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = batch_size,
     subset = "training",
     class_mode = "categorical",
     color_mode = "grayscale",
     seed = 100
)

# Cargar datos de validación
val_data_gen <- flow_images_from_directory(
     target_size = c(150L, 150L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = batch_size,
     subset = "validation",
     class_mode = "categorical",
     color_mode = "grayscale",
     seed = 100
)

# Capa de entrada explícita
input_layer <- layer_input(shape = c(150, 150, 1), name = "input_layer")

# Construcción del modelo usando el enfoque funcional
output_layer <- input_layer %>%
     layer_conv_2d(filters = 32L, kernel_size = c(3, 3), activation = "relu", name = "conv1") %>%
     layer_batch_normalization(name = "batch_norm1") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool1") %>%
     layer_dropout(rate = 0.3, name = "dropout1") %>%
     layer_conv_2d(filters = 64L, kernel_size = c(3, 3), activation = "relu", name = "conv2") %>%
     layer_batch_normalization(name = "batch_norm2") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool2") %>%
     layer_dropout(rate = 0.3, name = "dropout2") %>%
     layer_conv_2d(filters = 128L, kernel_size = c(3, 3), activation = "relu", name = "conv3") %>%
     layer_batch_normalization(name = "batch_norm3") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool3") %>%
     layer_flatten(name = "flatten") %>%
     layer_dense(units = 512L, activation = "relu", name = "dense1") %>%
     layer_dropout(rate = 0.5, name = "dropout3") %>%
     layer_dense(units = 5L, activation = "softmax", name = "output")

# Crear el modelo funcional
modelo <- keras_model(inputs = input_layer, outputs = output_layer)

modelo$compile(
     optimizer = optimizer_adam(learning_rate = 0.001),
     loss = "categorical_crossentropy", # Para clasificación multi-clase
     metrics = list("accuracy")
)

# Verificar el resumen del modelo
modelo$summary()

# Guardar el mejor modelo durante el entrenamiento
callbacks <- list(
     callback_model_checkpoint( 
          filepath = file.path(output_dir, "mejor_modelo.keras"), 
          monitor = "val_loss", 
          verbose = 0, 
          save_best_only = TRUE, 
          save_weights_only = FALSE, 
          mode = c("auto", "min", "max"), 
          period = NULL, 
          save_freq = "epoch" 
     )
)

# Número de muestras de entrenamiento y validación
train_samples <- train_data_gen$n
valid_samples <- val_data_gen$n

start_time <- Sys.time()

history <- modelo$fit(
     x = train_data_gen,
     steps_per_epoch = as.integer(floor(train_samples / batch_size)),
     epochs = 30L,
     validation_data = val_data_gen,
     validation_steps = as.integer(floor(valid_samples / batch_size)),
     callbacks = callbacks
)

end_time <- Sys.time()
duration <- end_time - start_time
print(duration)

modelo %>% evaluate(train_data_gen)

# Evaluación del Modelo B&N ####
true_classes <- val_data_gen$classes  # Asegúrate de que esto corresponda a la organización de tus datos
class_names <- c("H1", "H2", "H3", "H5", "H6")
class_pred <- modelo %>%
     predict(val_data_gen) %>%
     k_argmax() %>%
     as.numeric()

# Convertir índices a factores usando nombres de clase
predictions <- factor(class_names[class_pred + 1], levels = class_names)  # Suma 1 porque R es indexado en 1
true_labels <- as.factor(class_names[true_classes + 1])  # Suma 1 por la misma razón

# Crear la matriz de confusión usando yardstick
conf_mat_data <- yardstick::conf_mat(data = data.frame(Observado = true_labels, Estimado = predictions),
                          truth = Observado,
                          estimate = Estimado)

# Visualización de la matriz de confusión
autoplot(conf_mat_data, type = "heatmap") +
     theme(axis.text.x = element_text(angle = 20, vjust = 0.5, hjust=1),
           axis.title.x = element_text(vjust = -2))

# Modelo RGB ####

# Cargar datos de entrenamiento
train_data_gen2 <- flow_images_from_directory(
     target_size = c(500L, 500L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = 32,
     subset = "training",
     class_mode = "categorical",
     color_mode = "rgb",
     seed = 100
)

# Cargar datos de validación
val_data_gen2 <- flow_images_from_directory(
     target_size = c(500L, 500L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = 32,
     subset = "validation",
     class_mode = "categorical", # 5 clases
     color_mode = "rgb",
     seed = 100
)

# Capa de entrada explícita
input_layer2 <- layer_input(shape = c(500, 500, 3), name = "input_layer")

# Construcción del modelo usando el enfoque funcional
output_layer2 <- input_layer2 %>%
     layer_conv_2d(filters = 32L, kernel_size = c(3, 3), activation = "relu", name = "conv1") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool1") %>%
     layer_conv_2d(filters = 64L, kernel_size = c(3, 3), activation = "relu", name = "conv2") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool2") %>%
     layer_conv_2d(filters = 128L, kernel_size = c(3, 3), activation = "relu", name = "conv3") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool3") %>%
     layer_flatten(name = "flatten") %>%
     layer_dense(units = 512L, activation = "relu", name = "dense1") %>%
     layer_dense(units = 5L, activation = "softmax", name = "output")

# Crear el modelo funcional
modelo2 <- keras_model(inputs = input_layer2, outputs = output_layer2)

modelo2$compile(
     optimizer = tensorflow::tf$keras$optimizers$RMSprop(learning_rate = 0.001),
     loss = "categorical_crossentropy",
     metrics = list("accuracy")
)

# Verificar el resumen del modelo
modelo2$summary()

# Guardar el mejor modelo durante el entrenamiento
callbacks2 <- list(
     callback_model_checkpoint( 
          filepath = file.path(output_dir, "mejor_modelo2.keras"), 
          monitor = "val_loss", 
          verbose = 0, 
          save_best_only = TRUE, 
          save_weights_only = FALSE, 
          mode = c("auto", "min", "max"), 
          period = NULL, 
          save_freq = "epoch" 
     )
)

# Entrenamiento
start_time <- Sys.time()

history2 <- modelo2$fit(
     x = train_data_gen2,
     steps_per_epoch = as.integer(floor(train_samples / batch_size)),
     epochs = 30L,
     validation_data = val_data_gen2,
     validation_steps = as.integer(floor(valid_samples / batch_size)),
     callbacks = callbacks2
)

end_time <- Sys.time()
duration <- end_time - start_time
print(duration)

# Evaluación del Modelo RGB ####

true_classes <- val_data_gen2$classes  # Asegúrate de que esto corresponda a la organización de tus datos
class_names <- c("H1", "H2", "H3", "H5", "H6")
class_pred <- modelo %>% 
     predict(val_data_gen2) %>% 
     k_argmax() %>% 
     as.numeric()

# Convertir índices a factores usando nombres de clase
predictions <- factor(class_names[class_pred + 1], levels = class_names)  # Suma 1 porque R es indexado en 1
true_labels <- as.factor(class_names[true_classes + 1])  # Suma 1 por la misma razón

# Crear la matriz de confusión usando yardstick
conf_mat_data <- yardstick::conf_mat(data = data.frame(Observado = true_labels, Estimado = predictions),
                                     truth = Observado,
                                     estimate = Estimado)

# Visualización de la matriz de confusión
autoplot(conf_mat_data, type = "heatmap") +
     theme(axis.text.x = element_text(angle = 20, vjust = 0.5, hjust=1),
           axis.title.x = element_text(vjust = -2))


# Modelo B&N ####
# Generadores de datos
train_image_array_gen <- image_data_generator(
     rescale = 1 / 255,
     rotation_range = 30,
     width_shift_range = 0.2,
     height_shift_range = 0.2,
     shear_range = 0.2,
     zoom_range = 0.2,
     horizontal_flip = TRUE,
     fill_mode = "nearest",
     validation_split = 0.2
)

# Cargar datos de entrenamiento
batch_size = 32
train_data_gen <- flow_images_from_directory(
     target_size = c(150L, 150L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = batch_size,
     subset = "training",
     class_mode = "binary",
     color_mode = "grayscale",
     seed = 100
)

# Cargar datos de validación
val_data_gen <- flow_images_from_directory(
     target_size = c(150L, 150L),
     directory = output_dir,
     generator = train_image_array_gen,
     batch_size = batch_size,
     subset = "validation",
     class_mode = "binary",
     color_mode = "grayscale",
     seed = 100
)

# Capa de entrada explícita
input_layer <- layer_input(shape = c(150, 150, 1), name = "input_layer")

# Construcción del modelo usando el enfoque funcional
output_layer <- input_layer %>%
     layer_conv_2d(filters = 32L, kernel_size = c(3, 3), activation = "relu", name = "conv1") %>%
     layer_batch_normalization(name = "batch_norm1") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool1") %>%
     layer_dropout(rate = 0.3, name = "dropout1") %>%
     layer_conv_2d(filters = 64L, kernel_size = c(3, 3), activation = "relu", name = "conv2") %>%
     layer_batch_normalization(name = "batch_norm2") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool2") %>%
     layer_dropout(rate = 0.3, name = "dropout2") %>%
     layer_conv_2d(filters = 128L, kernel_size = c(3, 3), activation = "relu", name = "conv3") %>%
     layer_batch_normalization(name = "batch_norm3") %>%
     layer_max_pooling_2d(pool_size = c(2, 2), name = "pool3") %>%
     layer_flatten(name = "flatten") %>%
     layer_dense(units = 512L, activation = "relu", name = "dense1") %>%
     layer_dropout(rate = 0.5, name = "dropout3") %>%
     layer_dense(units = 1L, activation = "sigmoid", name = "output")

# Crear el modelo funcional
modelo <- keras_model(inputs = input_layer, outputs = output_layer)

modelo$compile(
     optimizer = optimizer_adam(learning_rate = 0.001),
     loss = "binary_crossentropy",
     metrics = list("accuracy")
)

# Verificar el resumen del modelo
modelo$summary()

# Guardar el mejor modelo durante el entrenamiento
callbacks <- list(
     callback_model_checkpoint( 
          filepath = file.path(output_dir, "mejor_modelo.keras"), 
          monitor = "val_loss", 
          verbose = 0, 
          save_best_only = TRUE, 
          save_weights_only = FALSE, 
          mode = c("auto", "min", "max"), 
          period = NULL, 
          save_freq = "epoch" 
     )
)

# Número de muestras de entrenamiento y validación
train_samples <- train_data_gen$n
valid_samples <- val_data_gen$n

start_time <- Sys.time()

history <- modelo$fit(
     x = train_data_gen,
     steps_per_epoch = as.integer(floor(train_samples / batch_size)),
     epochs = 30L,
     validation_data = val_data_gen,
     validation_steps = as.integer(floor(valid_samples / batch_size)),
     callbacks = callbacks
)

end_time <- Sys.time()
duration <- end_time - start_time
print(duration)

modelo %>% evaluate(train_data_gen)

# Evaluación del Modelo B&N ####
true_classes <- val_data_gen$classes  # Asegúrate de que esto corresponda a la organización de tus datos
class_names <- c("H1", "No H1")
class_pred <- modelo %>%
     predict(val_data_gen) %>%
     k_argmax() %>%
     as.numeric()

# Convertir índices a factores usando nombres de clase
predictions <- factor(class_names[class_pred + 1], levels = class_names)  # Suma 1 porque R es indexado en 1
true_labels <- as.factor(class_names[true_classes + 1])  # Suma 1 por la misma razón

# Crear la matriz de confusión usando yardstick
conf_mat_data <- yardstick::conf_mat(data = data.frame(Observado = true_labels, Estimado = predictions),
                                     truth = Observado,
                                     estimate = Estimado)

# Visualización de la matriz de confusión
autoplot(conf_mat_data, type = "heatmap") +
     theme(axis.text.x = element_text(angle = 20, vjust = 0.5, hjust=1),
           axis.title.x = element_text(vjust = -2))
