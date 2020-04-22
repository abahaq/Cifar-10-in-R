install.packages('keras')
install.packages('tenserflow')
tensorflow::install_tensorflow()
install_keras()


library(keras)
library(dplyr)
library(tensorflow)

version
library(ggplot2)
reticulate::install_miniconda()
reticulate::py_config()



df <-  dataset_cifar10()


#Train & Test Split

c(train_images, train_labels) %<-% df$train
c(test_images, test_labels) %<-% df$test

class_names <- c('airplane',
                 'automobile',
                 'bird',
                 'cat',
                 'deer',
                 'dog',
                 'frog',
                 'horse',
                 'ship',
                 'truck')

### Preporcess Data ###
train_images <- train_images / 255
test_images <- test_images / 255

val_indices <- 1:5000
val_images <-  train_images[val_indices,,,]
part_train_images <- train_images[-val_indices,,,]
val_labels <- train_labels[val_indices]
part_train_labels <- train_labels[5001:50000]

part_train_images %>% glimpse()
part_train_images <- array_reshape(part_train_images,c(45000, 32, 32, 3))
val_images <- array_reshape(val_images, c(5000, 32, 32, 3))
test_images <- array_reshape(test_images, c(10000, 32, 32, 3))
 
####  Define Model Architecture   ####

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(32, 32, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2))

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 300, activation = 'relu') %>% 
  layer_dense(units = 300, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

model

#####     Configuring the Model     #####

model %>% compile(
  optimizer = 'sgd',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

## sparse_categorical_crossentropy => more than 2 classes and observation can belong to only one class
## Binary_crossentropy => 2 classes and object belongs to one of the two classes
## Categorical_crossentropy => more than 2 classes and observation can belong to multiple classes

model %>% fit(part_train_images, 
              part_train_labels, 
              epochs = 30, 
              batch_size = 30, 
              validation_data=list(val_images,val_labels))

CNN_score <- model %>% evaluate(test_images,test_labels)

cat("Test loss:", CNN_score$loss,"\n")
cat("Test accuracy:", CNN_score$acc, "\n")

######     Predicting on Test Set     ######

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
class_names[class_pred[1:20]+1]
class_names[test_labels[1:20]+1]
plot(as.raster(test_images[1, , , ]))  #plot(as.raster(test_images[1, , , ]), max = 255)

















