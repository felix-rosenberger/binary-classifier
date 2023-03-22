# Binary Classifier

This repository contains the implementation of a binary classifier using logistic regression, a cross-entropy loss function, and gradient descent as a learning algorithm. The classifier is written in R and is compared to a built-in logistic regression classifier provided by the `glm` function.

## Files

There are three main files in this repository:

1. [`binary_classifier.R`](https://github.com/felix-rosenberger/binary-classifier/blob/main/binary_classifier.R) : Contains the implementation of the binary classifier, including the training and prediction functions, and a function to visualise the loss function for different learning rates and to determine an optimal learning rate.
2. [`binary_classifier_example.R`](https://github.com/felix-rosenberger/binary-classifier/blob/main/binary_classifier_example.R): A script that demonstrates how to use the binary classifier, and compares its performance to the built-in `glm` classifier.
3. `BinaryClassifier.csv`: The dataset used in the example script.

## Dependencies

The following R libraries are used in the scripts:

- `tidyverse`
- `caret`
- `ggplot2`
- `reshape2`
- `pacman`

Please make sure to install these libraries before running the scripts.

## Usage

1. Clone the repository to your local machine.
2. Open R or RStudio and set the working directory to the repository folder.
3. Install the required libraries if you haven't already.
4. Run the `binary_classifier_example.R` script to see the binary classifier in action, and compare its performance to the built-in `glm` classifier.

## Classifier Implementation

The `binary_classifier.R` file contains two main functions:

- `binary.classifier.train()`: A function for training the binary classifier on a given dataset.
- `binary.classifier.predict()`: A function for making predictions using the trained binary classifier on a test dataset.

### Function Inputs and Outputs

#### binary.classifier.train()

Inputs:
- `train_data`: A data frame with the training data, where the column name of the binary target variable should be "y".
- `learning_rate`: The learning rate for the gradient descent algorithm (a numeric value).
- `iterations`: The number of iterations for the gradient descent algorithm (an integer value).

Outputs:
- A list including the following attributes:
  - `weights`: A numeric vector containing the weights of the trained classifier.
  - `loss_history`: A numeric vector containing the loss history for each iteration.

#### binary.classifier.predict()

Inputs:
- `test_data`: A data frame with the test data, where the target variable column should be named "y".
- `trained_classifier`: A trained classifier object (the output from `binary.classifier.train()`).

Outputs:
- A list including the following attributes:
  - `pred.classes`: A numeric vector containing the class predictions for the test data.
  - `conf.matrix`: A confusion matrix with all relevant metrics to evaluate the classifier performance.

## Example Script

The `binary_classifier_example.R` script demonstrates the usage of the custom binary classifier and compares its performance to the built-in `glm` classifier. It includes the following steps:

1. Loading the required libraries and the custom classifier functions.
2. Reading and preparing the dataset.
3. Creating train and test data.
4. Fitting the `glm` classifier and making predictions on the test data.
5. Training the custom binary classifier for different learning rates.
6. Plotting the loss functions for different learning rates.
7. Training and predicting with the optimal learning rate using the custom binary classifier.
8. Comparing the performance of the custom classifier to the built-in `glm` classifier.

For more details, please refer to the comments within the `binary_classifier_example.R` script.
