import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(num_features):
    """
	Initialize the parameters to zero (weights and offset)

	parameters:
    num_features: the number of features(columns) in the train data (to define the size of the weights vector)

	returns: 
	w: vector of weights with the dimension of num_features and 1
	b: the offset term (scalar)
	"""
    w = np.zeros((num_features,1)) 
    b = 0
    return w, b

def model_forward(x, w, b):
    """
    Perform the forward propagation step of linear regression
    
    parameters:
    x: the input data
    w: weights vector
    b: the offset term (scalar)

    returns:
    z: predictions of shape number of inputs and 1, result of the linear model
    """
    z = np.dot(x,w) + b # calculate the predicted value using multiplication of the inputs x and weights w, adding the offset b
    return z

def compute_cost(real_values,predictions):
    """
    Compute the Mean Squared Error cost between the real and predicted values.
    
    parameters:
    real_values: the actual target values.
    predictions: the model's predicted values

    returns:
    cost: the computed Mean Squared Error cost
    """
    n = real_values.shape[0] # the number of examples
    loss = (real_values - predictions) ** 2 # calculate the loss (error squared) for each example
    loss_total = np.sum(loss) # sum all the losses
    cost = loss_total / n # calculate the average loss
    return cost

def model_backward(inputs, real_values, predictions):
    """
    Compute the gradients of the cost function with respect to the weights and offset.

    parameters:
    inputs: the inputs vector
    real_values: the actual target values
    predictions: the model's predicted values

    returns:
    gradient_w: the gradient of the cost function with respect to the weights
    gradient_b: the gradient of the cost function with respect to the offset
    """
    n = inputs.shape[0] # the number of training examples
    error = predictions - real_values # calculate error for each example
    gradient_w = (2/n) * np.dot(inputs.T, error) # calculate gradient with respect to weights
    gradient_b = (2/n) * np.sum(error) # calculate gradient with respect to offset
    return gradient_w, gradient_b

def update_parameters(learning_rate, w, b, gradient_w, gradient_b): 
    """
    Update the model's weights and offset based on the calculated gradients

    parameters:
    learning_rate: the step size at each iteration while moving toward a minimum of the cost function
    w: current weights of the model
    b: current offset of the model
    gradient_w: gradient of the cost function with respect to the weights
    gradient_b: gradient of the cost function with respect to the offset

    returns:
    w: updated weights after applying the gradient descent step
    b: updated offset after applying the gradient descent step
    """
    w = w - (learning_rate * gradient_w) # update weights by subtracting the product of learning rate and gradient with respect to weights from current weights
    b = b - (learning_rate * gradient_b) #update offset by subtracting the product of learning rate and gradient with respect to offset from current offset
    return w, b

def predict(inputs,w,b):
    """
    Predict the target values for a given set of input features using the linear model parameters.

    parameters:
    inputs: the inputs vector
    w: weights of the linear model
    b: offset of the linear model

    returns:
    predictions: predicted values
    """
    predictions = model_forward(inputs, w, b) # calculate predictions using the linear model's forward propagation
    return predictions

def train_linear_model(inputs, target, num_iterations, learning_rate):
    """
    Train a linear regression model using gradient descent.

    This function iteratively updates the model's weights and offset to minimize the cost function. It performs model_forward and model_backward functions, computes the cost, and updates the model parameters based on the computed gradients. It also tracks and prints the cost at specified intervals to monitor the training process.

    parameters:
    inputs: the inputs of the training data
    target: the target values ('mpg' column)
    num_iterations: the number of iterations for which to train the model
    learning_rate: the learning rate to use for parameter updates during training

    returns:
    w: the final optimized weights of the model
    b: the final optimized offset values of the model
    cost_list: the list of costs calculated at specified intervals during training
    iterations: the list of iteration numbers at which costs were recorded
    """
    num_features = inputs.shape[1] # the number of inputs
    w, b = initialize_parameters(num_features) # initialize parameters
    cost_list = []
    iterations = []
    
    for iteration in range(num_iterations):  
        predictions = model_forward(inputs,w,b) # forward pass to get predictions
        iteration_cost = compute_cost(target,predictions) # compute the cost between target values and predictions

        # record and print the cost and iteration number at specified intervals and at the last iteration
        if iteration % 200 == 0 or iteration == num_iterations - 1:
            cost_list.append(iteration_cost)
            iterations.append(iteration)
            print(f"The cost for iteration {iteration} is {iteration_cost}")

        gradient_w, gradient_b = model_backward(inputs, target, predictions) # backward pass to compute gradients
        w, b = update_parameters(learning_rate, w, b, gradient_w, gradient_b) # update model parameters using the computed gradients
    return w, b, cost_list, iterations


def show_plot(costs, learning_rates):
    """
    Display a plot comparing the cost function's value over iterations for different learning rates

    parameters:
    costs: list of costs recorded at specified intervals during training for a particular learning rate.
    learning_rates: list of learning rates used during training

    returns:
    the function shows a plot
    """
    iterations = [0,200,400,600,800,1000] # fixed iterations where costs are recorded (x-axis)

    [plt.plot(iterations, cost, label=f'α={rate}') for rate, cost in zip(learning_rates, costs)] # create a plot for each set of costs associated with a specific learning rate
        
    plt.ylabel('Cost') # set the y-axis label
    plt.xlabel('Iteration') # set the x-axis label
    plt.legend() # show the plot's legend
    plt.show()