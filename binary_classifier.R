# learning function
binary.classifier.train = function(
    train.df, # training dataframe, labels have to be named "y"
    theta = rep(1, ncol(train)), # vector of weights dependent on training data size
    alpha, # numerical value, learning rate for optimisation
    epsilon = 0.000001, # convergence tolerance for optimisation
    iter.max = 1000) { # maximum number of iterations for optimisation
  
  # check for correct column name of target variable
  
  # preprocess input data
  X = train[, !colnames(train) %in% c("y", "Y", "labels", "Labels", "classes", "Classes")] # select features only
  X = scale(X) # standardise features
  col.means = attr(X,"scaled:center") # save feature means for test data standardisation
  col.std = attr(X,"scaled:scale") # save feature standard deviation for test data
  X = as.matrix( # convert final output of training data to matrix
    cbind( # combine bias term and training data
      rep(1, nrow(train)), X)) # create sequence of bias terms and combine with scaled data
  y = matrix(train[, colnames(train) %in% c("y", "Y", "labels", "Labels", "classes", "Classes")], ncol = 1) # create label vector
  
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
                learning.rate = alpha,
                max.iter = iter.max) # store for loss function visualisations
  return(result)
  
}

# prediction function
binary.classifier.predict = function(test.df, # test dataframe, labels have to be named "y"
                                     trained.classifier, # trained classifier object
                                     threshold = 0.5) { # set prediction threshold
  # preprocess input data
  X = test[, !colnames(test) %in% c("y", "Y", "labels", "Labels", "classes", "Classes")]
  # standardise each feature column in test data (mean and std estimates from training data)
  for (i in 1:ncol(X)) {
    X[,i] = scale(X[,i], 
                  center = trained.classifier$train.means[i], 
                  scale = trained.classifier$train.std[i])
  }
  X = as.matrix( # convert final output of test data to matrix
    cbind(rep(1, nrow(test)), X)) # create sequence of bias terms and combine with scaled data
  y = matrix(test[, colnames(train) %in% c("y", "Y", "labels", "Labels", "classes", "Classes")], ncol = 1)
  
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
  if (!require("pacman")) install.packages("pacman") # checks for packages; installs if required
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

# loss function visualisation and learning rate determination
loss.function.visualisation = function(min.alpha, # smallest learning rate
                                       max.alpha, # highest learning rate
                                       increments, # increments for learning rate to train model
                                       max.iter = 1000){
  # train for different learning rates
  alphas = seq(min.alpha, max.alpha, by = increments) # define sequence of alphas
  costs = data.frame(matrix(NA, nrow = max.iter, 
                            ncol = length(alphas))) # create empty df with nrow = max.iter
  final.costs = c()
  init.iter = max.iter
  optimal.alpha = 0
  # for each alpha, train classifier and add costs as column to results df
  for (i in 1:length(alphas)) {
    classifier = binary.classifier.train(train, alpha = alphas[i], iter.max = max.iter)
    cost = c(classifier$cost) # costs history of classifier for each alpha
    final.costs = cbind(final.costs, classifier$final.cost)
    costs[,i] = c(cost, rep(NA, nrow(costs) - length(cost))) # append costs to results df
    names(costs)[i] = paste("alpha =", alphas[i]) # rename column to alpha level
    iter = classifier$iterations # extract iterations for alpha
    # update optimal learning rate based on convergence speed
    if (iter <= init.iter){
      optimal.alpha = alphas[i]
      init.iter = iter
    }
  }
  
  # plot loss functions with ggplot
  # change to long format
  suppressWarnings({
    costs.long = melt(costs, variable.name = "Alpha", value.name = "Cost")
  })
  costs.long$Iteration = sequence(rep(nrow(costs), ncol(costs))) # add iterations for each alpha
  costs.long = costs.long[!is.na(costs.long$Cost),] # remove rows with missing values
  
  # Create a data frame with the final iteration count for each alpha
  final.iterations = data.frame(Alpha = unique(costs.long$Alpha),
                                FinalIteration = tapply(costs.long$Iteration, costs.long$Alpha, 
                                                        max)) # selecting max iteration value per alpha
  best.alpha = final.iterations[order(final.iterations$FinalIteration), 
                                "Alpha"][1:3] # 3 best learning rates based on convergence speed
  max.iter = max(final.iterations$FinalIteration[final.iterations$Alpha %in% best.alpha]) # get maximum iterations from best alphas
  x.range = max.iter + 10 # set x range
  
  # find maximum cost at last iteration
  cost.final = merge(costs.long, final.iterations, by = "Alpha") # show max number of iterations on every row for each alpha
  costs.finaliter = subset(cost.final, Iteration == FinalIteration) # keep only those with max iteration to keep costs
  max.cost = max(costs.finaliter$Cost) 
  min.cost = min(costs.finaliter$Cost)
  # set y range
  y.max = max.cost + 10
  y.min = min.cost - 1
  
  # create plot with loss functions over learning rate range
  plot = ggplot(costs.long, aes(x = Iteration, y = Cost, color = Alpha)) +
    geom_line() +
    scale_color_discrete(name = "Learning Rate") +
    labs(title = "Loss Visualisation for Different Learning Rates",
         subtitle = "Binary Cross-Entropy Loss Function",
         x = "Iteration",
         y = "Cost") +
    coord_cartesian(xlim = c(0, x.range),
                    ylim = c(y.min, y.max)) + # set axis limit
    theme_minimal()
  
  # create result output
  result = list(loss.plot = plot,
                optimal.alpha = optimal.alpha)
  
  return(result)
}







