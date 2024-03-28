import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(num_features):
    weights = np.zeros((num_features,1))
    offset = 0
    return weights, offset

# function for evaluating model
def model_forward(x, w, b):
# prediction
# multiplication of the weights in x plus offset
    z = np.dot(x,w) + b
    return z

def compute_cost(real_values,predicted_values):
    n = real_values.shape[0]
    loss = (real_values - predicted_values) ** 2
    loss_sum = np.sum(loss)
    cost = loss_sum / n 
    return cost

def model_backward(features, real_values, predicted_values):
    n = features.shape[0]
    error = predicted_values - real_values
    partial_derivative_w = (2/n) * np.dot(features.T, error)
    partial_derivative_b = (2/n) * np.sum(error)
    return partial_derivative_w, partial_derivative_b

def update_parameters(learning_rate, weights, b, gradient_w, gradient_b): 
    weights = weights - (learning_rate * gradient_w)
    b = b - (learning_rate * gradient_b)
    return weights, b

def predict(features,w,b):
    predictions = model_forward(features, w, b)
    return predictions

#  final training function
def train_linear_model(inputs, target_column, num_iterations, learning_rate):
    num_features = inputs.shape[1]
    weights, b = initialize_parameters(num_features)
    cost_list = []
    iterations = []
    for iteration in range(num_iterations):
        predicted_values = model_forward(inputs,weights,b)
        iteration_cost = compute_cost(target_column,predicted_values)
        if iteration % 200 == 0 or iteration == num_iterations - 1:
            cost_list.append(iteration_cost)
            iterations.append(iteration)
            print(f"The cost for iteration {iteration} is {iteration_cost}")
        gradient_w, gradient_b = model_backward(inputs, target_column, predicted_values)
        weights, b = update_parameters(learning_rate, weights, b, gradient_w, gradient_b)
    return weights, b, cost_list, iterations


def show_plot(costs, learning_rates):
    # set figure size
    plt.figure(figsize=(10,10))
    iterations = [0,200,400,600,800,1000]

    [plt.plot(iterations, cost, label=f'α={rate}') for rate, cost in zip(learning_rates, costs)]
        
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()