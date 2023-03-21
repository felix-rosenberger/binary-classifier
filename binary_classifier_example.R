# first load required packages
library(tidyverse)
library(caret)

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

binary.classifier.train = function(
    train.df, # training dataframe, labels have to be named "y"
    theta = rep(1, ncol(train)), # vector of weights dependent on training data size
    alpha, # numerical value, learning rate for optimisation
    epsilon = 0.000001, # convergence tolerance for optimisation
    iter.max = 1000) { # maximum number of iterations for optimisation
  
  # preprocess input data
  X = train[, colnames(train) != "y"] # select features only
  X = scale(X) # standardise features
  col.means = attr(X,"scaled:center") # save feature means for test data standardisation
  col.std = attr(X,"scaled:scale") # save feature standard deviation for test data
  X = as.matrix( # convert final output of training data to matrix
    cbind( # combine bias term and training data
      rep(1, nrow(train)), X), # create sequence of bias terms and combine with scaled data
    ncol = ncol(train))
  y = matrix(train[, colnames(train) == "y"], ncol = 1) # create label vector
  
  # initialise parameters
  res.epsilon = 1 # tolerance
  res.theta = theta # parameter weights
  iter = 1 # iteration
  theta = matrix(theta, ncol = 1) # transform vector into matrix
  
  # cost function definition
  y_hat = 1 / (1 + exp(-(X %*% theta))) # define sigmoid function for prediction 
  res.cost = -sum(y * log(y_hat) + (1 - y) * log(1 - y_hat)) # define binary CE loss function and initial loss
  
  # model training
  while (res.epsilon > epsilon & iter < iter.max) { 
    y_hat = 1 / (1 + exp(-(X%*%theta))) # define again for each prediction
    theta.new = theta - alpha * (t(X) %*% (y_hat - y)) # determine new theta using CE gradient and learning rate
    res.theta = cbind(res.theta, theta.new) # for each iteration, add new weights
    cost = -sum(y * log(y_hat) + (1 - y) * log(1 - y_hat)) # determine cost for current weights
    res.cost = cbind(res.cost, cost) # for each iteration, add cost for respective weights
    res.epsilon = sum((theta-theta.new)**2)^0.5 # determine tolerance at each iteration and update
    theta = theta.new # update weights for next iteration
    iter = iter + 1 # update counter for iterations
  }
  #create result list for later extraction
  result = list(final.weight = theta, 
                weights = res.theta, 
                final.cost = cost,
                cost = res.cost,
                iterations = iter,
                tolerance = res.epsilon,
                train.means = col.means,
                train.std = col.std,
                learning.rate = alpha)
  return(result)
  
}

binary.classifier.predict = function(test.df, # test dataframe, labels have to be named "y"
                                     trained.classifier, # trained classifier object
                                     threshold = 0.5) { # set prediction threshold
  # preprocess input data
  X = test[, colnames(test) != "y"]
  # standardise each feature column in test data (mean and std estimates from training data)
  for (i in 1:ncol(X)) {
    X[,i] = scale(X[,i], 
                  center = trained.classifier$train.means[i], 
                  scale = trained.classifier$train.std[i])
  }
  X = as.matrix( # convert final output of test data to matrix
    cbind( # combine bias term and test data
      rep(1, nrow(test)), X), # create sequence of bias terms and combine with scaled data
    ncol = ncol(test))
  y = matrix(test[, colnames(train) == "y"], ncol = 1)
  
  # predict output based on optimal weights
  theta = trained.classifier$final.weight
  y.pred = 1 / (1 + exp(-(X%*%theta)))
  
  # convert probabilities into classes
  y.class = matrix(NA, nrow = nrow(y.pred), ncol = 1)
  for (i in 1:nrow(y.pred)) {
    if (y.pred[i] > threshold) {
      y.class[i] = 1
    }
    else {
      y.class[i] = 0
    }
  }
  
  # model evaluation
  if (!require("pacman")) install.packages("pacman") # wrapper, checks if packages are installed; 
  #installs if required
  pacman::p_load(caret)
  
  # confusion matrix for later retrieval
  y.true = test.df[, colnames(test) == "y"] # save true labels
  conf.matrix = confusionMatrix(factor(y.class),factor(y.true), mode = "everything")
  
  result = list(pred.probabilities = y.pred, # save probabilities for each test observation
                pred.classes = y.class, # save predicted class for each test observation
                true.classes = y.true, # save true classes for retrieval and comparison
                conf.matrix = conf.matrix, # save confusion matrix
                weights = theta) # save weights used
  return(result)
}

# train for different alphas
alphas = seq(0.01, 0.1, by = 0.02) # define sequence of alphas
costs = data.frame(matrix(NA, nrow = 1000, ncol = length(alphas))) # create empty df with nrow = max.iter
final.costs = c()

# for each alpha, train classifier and add costs as column to results df
for (i in 1:length(alphas)) {
  classifier = binary.classifier.train(train, alpha = alphas[i])
  cost = c(classifier$cost) # costs for each alpha
  final.costs = cbind(final.costs, classifier$final.cost)
  costs[,i] = c(cost, rep(NA, nrow(costs) - length(cost))) # append costs to results df
  names(costs)[i] = paste("alpha =", alphas[i]) # rename column to alpha level
}

# for first alpha
plot(costs[,1], ylab="Cost", xlab="Iteration", type = "b", col = "black",
     main=names(costs)[1])
abline(h=final.costs[1],col="red")
legend(700, 250, legend=c(paste("Final Cost: ", round(final.costs[1], 0)), "Loss Function"),
       col=c("red", "black"), lty=1:2, cex=0.8,
       box.lty=0)
# for second alpha
plot(costs[,2], ylab="Cost", xlab="Iteration", type = "b", col = "black",
     main=names(costs)[2])
abline(h=final.costs[2],col="red")
legend(700, 250, legend=c(paste("Final Cost: ", round(final.costs[2], 0)), "Loss Function"),
       col=c("red", "black"), lty=1:2, cex=0.8,
       box.lty=0)
# for third alpha
plot(costs[,3], ylab="Cost", xlab="Iteration", type = "b", col = "black",
     main=names(costs)[3])
abline(h=final.costs[3],col="red")
legend(700, 300, legend=c(paste("Final Cost: ", round(final.costs[3], 0)), "Loss Function"),
       col=c("red", "black"), lty=1:2, cex=0.8,
       box.lty=0)
# for fourth alpha
plot(costs[,4], ylab="Cost", xlab="Iteration", type = "b", col = "black",
     main=names(costs)[4])
abline(h=final.costs[4],col="red")
legend(700, 450, legend=c(paste("Final Cost: ", round(final.costs[4], 0)), "Loss Function"),
       col=c("red", "black"), lty=1:2, cex=0.8,
       box.lty=0)
# for fifth alpha
plot(costs[,5], ylab="Cost", xlab="Iteration", type = "b", col = "black",
     main=names(costs)[5])
abline(h=final.costs[5],col="red")
legend(700, 600, legend=c(paste("Final Cost: ", round(final.costs[5], 0)), "Loss Function"),
       col=c("red", "black"), lty=1:2, cex=0.8,
       box.lty=0)

# training with optimal alpha value
binary.train = binary.classifier.train(train, alpha = 0.03)

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
















