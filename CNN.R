library(keras)
library(tensorflow)
library(pROC)
reticulate::use_python('C:/ProgramData/Anaconda3/python.exe')
reticulate::py_config() # tensorflow version 2.12.0

## Data preparation and generator configuration

# directory of the images
train_dir <- "data/train"
validation_dir <- "data/validation"

# normalize the RGB values from [0-255] into [0-1]
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

# set train and validation generator
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(64, 64),
  batch_size = 20, # 100 batches in total
  class_mode = "categorical"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(64, 64),
  batch_size = 20, # same batch_size to train_generator, 40 batches in total
  class_mode = "categorical"
)

# set the data augmentation generator
data_augment <- image_data_generator(
  rescale = 1/255,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  brightness_range = c(0.3, 1),
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# train data generator with data augmentation
train_generator_aug <- flow_images_from_directory(
  train_dir,
  data_augment,
  target_size = c(64, 64),
  batch_size = 20, # same batch_size to train_generator
  class_mode = "categorical"
)

## Model training and validation

# deploy a convolutional neural network with data augmentation
model <- keras_model_sequential() %>%
  # convolutional layers
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # fully connected layers
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax") %>%
  # compile
  compile(
    loss = "categorical_crossentropy",
    metrics = "accuracy",
    optimizer = optimizer_adam()
  )

# fit the CNN model with data augmentation
fit <- model %>% fit(
  train_generator_aug,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 40, # no more than the batches of validation_generator
  verbose = 0
)

# Save the model
save_model_tf(model, "model/") # no non-English letters in the path

## Training and predictive performance visualization

# check accuracy and loss learning curve
out <- cbind(fit$metrics$accuracy,
                 fit$metrics$val_accuracy,
                 fit$metrics$loss,
                 fit$metrics$val_loss)
cols <- c("black", "dodgerblue3", "darkorchid4", "magenta")
# accuracy plot
matplot(out[,1:2],
        pch = 19, ylab = "Accuracy", xlab = "Epochs",
        col = adjustcolor(cols[1:2], 0.3),
        log = "y")
title("training and predictive performance")
# to add a smooth line to points
smooth_line <- function(y, span = 0.3) {
  x <- 1:length(y)
  out <- predict( loess(y ~ x, span = span) )
  return(out)
}
matlines(apply(out[,1:2], 2, smooth_line), lty = 1, col = cols[1:2], lwd = 2)
legend("topleft", 
       legend = c("Training", "Valid"),
       fill = cols[1:2], bty = "n")
# loss plot
matplot(out[,3:4], pch = 19, ylab = "Loss", xlab = "Epochs",
        col = adjustcolor(cols[3:4], 0.3))
title("training and predictive performance")
matlines(apply(out[,3:4], 2, smooth_line), lty = 1, col = cols[3:4], lwd = 2)
legend("topright", 
       legend = c("Training", "Valid"),
       fill = cols[3:4], bty = "n")

## Testing performance

# directory of the images
test_dir <- "data/test"
# normalize the RGB values from [0-255] into [0-1]
test_datagen <- image_data_generator(rescale = 1/255)
# set test generator
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(64, 64),
  batch_size = 20,
  class_mode = "categorical"
)
# to convert image to array
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(64, 64), 
                      grayscale = F # Set FALSE if image is RGB
    )
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255 # rescale image pixel
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
# convert image to array
x_test <- image_prep(test_generator$filepaths)
# one-hot encoding of target variable
y_test <- to_categorical(test_generator$labels)
# test performance
model %>% evaluate(x_test, y_test, verbose = 0)
# look at classification table
class_labels <- names(test_generator$class_indices)
class_y <- class_labels[max.col(y_test)]
class_hat_idx <- model %>% predict( x_test, verbose = 0) %>% max.col()
class_hat <- class_labels[class_hat_idx]

# Confusion Matrix
tab <- as.matrix(table(Actual = class_y, Predicted = class_hat))
tab
# overall accuracy
accuracy <- sum(diag(tab))/sum(tab)
accuracy
# precision
precision <- diag(tab)/colSums(tab)
# class-specific accuracy (recall)
recall <- diag(tab)/rowSums(tab)
# F-1 score
f1 = 2 * precision * recall / (precision + recall)
data.frame(precision, recall, f1)
# average auc score
auc <- multiclass.roc(class_y, class_hat_idx)
auc$auc
# roc curve
roc(class_y, class_hat_idx, plot = TRUE, legacy.axes = TRUE,
               levels=c('Border_collie', 'French_bulldog'), percent = TRUE, 
    main = "ROC curve for Border_collie and French_bulldog",
    xlab = "False positive rate",
    ylab = "True positive rate", col = "#377eb8", lwd = 2,
    print.auc = TRUE)
# find the 1st, 2nd and 3rd most likely class
class_labels_ordered <- sort(class_labels)
ranks <- t( apply(tab, 1, function(x) {
  class_labels_ordered[ order(x, decreasing = TRUE)[1:3]] } ) )
ranks


