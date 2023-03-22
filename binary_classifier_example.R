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

# train for different learning rates
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

# alternatively make nice plot with ggplot
# change to long format
costs.long = melt(costs, variable.name = "Alpha", value.name = "Cost")
costs.long$Iteration = sequence(rep(nrow(costs), ncol(costs))) # add iterations for each alpha

# remove rows with missing values
costs.long = costs.long[!is.na(costs.long$Cost),]

# Create a data frame with the final iteration count for each alpha
final.iterations = data.frame(Alpha = unique(costs.long$Alpha),
                               FinalIteration = tapply(costs.long$Iteration, costs.long$Alpha, 
                                                       max)) # selecting max iteration value per alpha

# find three best learning rates based on final costs
best.alpha = final.iterations[order(final.iterations$FinalIteration), "Alpha"][1:3]

# max number of iterations among three best rates
max.iter = max(final.iterations$FinalIteration[final.iterations$Alpha %in% best.alpha])

# set x range
x.range = max.iter + 10

# find maximum cost at last iteration
cost.final = merge(costs.long, final.iterations, by = "Alpha")
costs.finaliter = subset(cost.final, Iteration == FinalIteration)
max.cost = max(costs.finaliter$Cost)
min.cost = min(costs.finaliter$Cost)
y.max = max.cost + 10
y.min = min.cost - 10

# create plot with loss functions over learning rate range
ggplot(costs.long, aes(x = Iteration, y = Cost, color = Alpha)) +
  #geom_point() +
  geom_line() +
  scale_color_discrete(name = "Learning Rate") +
  labs(title = "Loss Functions for Different Learning Rates",
       subtitle = "Binary Cross-Entropy Loss",
       x = "Iteration",
       y = "Cost") +
  coord_cartesian(xlim = c(0, x.range),
                  ylim = c(y.min, y.max)) + # set axis limit
  theme_minimal()

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
















