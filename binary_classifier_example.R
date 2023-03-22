# first load required packages
library(tidyverse)
library(caret)
library(ggplot2)
library(reshape2)
source("binary_classifier.R") # load file containing the classifier functions

# read data
data = read.csv("BinaryClassifier.csv", header = FALSE)
names(data) = c("x1", "x2", "y") # assign variable names
head(data)

# create train and test data
train = as.data.frame(data[1:456, ])
test = as.data.frame(data[457:nrow(data), ])

# show comparison of statistical distribution of classes
freq = data.frame(y = c(0,1),
                  Train_frequency = c(sum(train$y == 0),
                                      sum(train$y == 1)),
                  Test_frequency = c(sum(test$y == 0),
                                     sum(test$y == 1)))
train_prop = freq$Train_frequency[2]/(freq$Train_frequency[1]+freq$Train_frequency[2]) * 100
test_prop = freq$Test_frequency[2]/(freq$Test_frequency[1]+freq$Test_frequency[2]) * 100
freq_comparison = cbind(train_prop, test_prop)
freq_comparison

# show plot with comparison
barplot(freq_comparison,
        main = "Proportional prevalence of classes across train and test",
        names.arg = c("Train data", "Test data"),
        ylab = "%",
        col = "darkblue",
        ylim = c(0,100))

# fit a model on the training data using the logit function for logistic regression
lr = glm(y ~., binomial(link = "logit"), data = train)
summary(lr)

# predict on test data
probabilities = lr %>% predict(test, type = "response")
predicted.classes = ifelse(probabilities > 0.5, 1, 0)
head(predicted.classes)

# confusion matrix
confusionMatrix(factor(predicted.classes),factor(test$y), mode = "everything")

# check classifier learning rates
loss.visualisation = loss.function.visualisation(min.alpha = 0.01,
                                                 max.alpha = 0.1,
                                                 increments = 0.01,
                                                 max.iter = 1000)

# training with optimal alpha value
binary.train = binary.classifier.train(train, alpha = loss.visualisation$optimal.alpha)

# make predictions on test set
binary.predict = binary.classifier.predict(test, binary.train)

# retrieve confusion matrix
binary.predict$conf.matrix

# scale training
train.scaled = as.data.frame(
  cbind(
    scale(train)[, colnames(train) != "y"], 
    train[, colnames(train) == "y"]))
names(train.scaled) = c("x1", "x2", "y")
col.means = attr(scale(train[, colnames(train) != "y"]),"scaled:center") 
col.std = attr(scale(train[, colnames(train) != "y"]),"scaled:scale")

# scale test
test.scaled = test[, colnames(train) != "y"]
for (i in 1:ncol(test.scaled)) {
  test.scaled[,i] = scale(test.scaled[,i], 
                          center = col.means[i], 
                          scale = col.std[i])
}

test.scaled = as.data.frame(cbind(test.scaled, test[, colnames(train) == "y"]))

# run glm again
lr.scaled = glm(y ~., binomial(link = "logit"), data = train.scaled)

# show table with coefficients to compare
comp.table = data.frame("glm classifier" = lr.scaled$coefficients,
                        "binary.classifier" = binary.train$final.weight)
knitr::kable(comp.table, 
             caption = "Model Estimates comparing built-in GLM methods and own Binary Classifier")
















