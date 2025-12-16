import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Sigmoid activation function
def sigmoid(z):
    # Clip z to prevent numerical overflow/underflow in np.exp for very large/small numbers
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Input data (x-values)
x = np.linspace(-10, 10, 500)
# Target curve (the complex wave we want to fit)
target = np.sin(x) * 0.5 + 0.5 * np.cos(2 * x) + 0.2

# Neural Network Architecture Parameters
n_neurons = 150  # Number of hidden "sigmoid" neurons

# Initialize NN Parameters randomly (weights and biases)
np.random.seed(0)

# 'slopes' act as weights connecting input 'x' to each hidden neuron (determining steepness)
slopes = np.random.uniform(-4, 4, n_neurons) 

# 'shifts' act implicitly as biases for each hidden neuron (determining horizontal position)
# More precisely, for a hidden neuron i, its input is z_i = slopes[i] * x - (slopes[i] * shifts[i]).
# So, -(slopes[i] * shifts[i]) acts as the bias for the i-th hidden neuron.
shifts = np.random.uniform(-10, 10, n_neurons)

# 'amps' act as weights connecting each hidden sigmoid neuron's output to the final output neuron
amps = np.random.uniform(-2, 2, n_neurons)

# Global bias for the output layer - allows the entire curve to shift up/down
output_bias = np.random.uniform(-0.5, 0.5)

# Learning rate for gradient descent
lr = 0.015 # Slightly adjusted for stability with more neurons and iterations

# Prepare the plot for animation
fig, ax = plt.subplots(figsize=(14, 8)) # Larger figure for better visualization

# Lines for individual sigmoid outputs (hidden layer activations)
# Reduced alpha and thinner lines to prevent clutter with many neurons
lines_sigmoids = [ax.plot([], [], lw=0.7, alpha=0.15, color='blue')[0] for _ in range(n_neurons)] 

# Line for the combined output of the NN (sum of weighted sigmoids + bias)
line_combined, = ax.plot([], [], lw=3, color='red', label='Combined NN Output') 

# Line for the target data
line_target, = ax.plot(x, target, 'k--', label='Target Data', alpha=0.9)

ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Neural Network Fitting Complex Shape (Learning in Progress)")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.legend()
ax.grid(True)

# Update function for the animation (one step of gradient descent)
def update(frame):
    global slopes, shifts, amps, output_bias

    # --- Forward Pass of the Neural Network ---
    
    # Calculate outputs of hidden sigmoid neurons
    # Each 'output' in this list corresponds to an h_i = sigmoid(slopes[i] * (x - shifts[i]))
    hidden_outputs = []
    for i in range(n_neurons):
        z_i = slopes[i] * (x - shifts[i]) # Input to sigmoid for neuron i
        hidden_outputs.append(amps[i] * sigmoid(z_i)) # Weighted output of neuron i

    # Sum hidden outputs and add the global output bias to get the final NN output
    combined_output = np.sum(hidden_outputs, axis=0) + output_bias

    # --- Backpropagation and Gradient Descent Updates ---

    # Calculate the overall error
    error = combined_output - target
    
    # Calculate Mean Squared Error for display
    mse = np.mean(error**2)

    # Gradients for hidden layer parameters (slopes, shifts, amps)
    for i in range(n_neurons):
        s = slopes[i]
        sh = shifts[i]
        a = amps[i]
        
        # Intermediate value for sigmoid derivative
        z_i = np.clip(s * (x - sh), -500, 500) # Clipped z_i for stable sigmoid derivative
        y_i = sigmoid(z_i) # Output of this specific sigmoid neuron

        # Derivative of sigmoid_i with respect to its input z_i
        # d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))
        dy_dz_i = y_i * (1 - y_i)

        # Gradients for the parameters of the i-th sigmoid
        d_error_ds = error * (a * dy_dz_i * (x - sh))
        d_error_dsh = error * (-a * dy_dz_i * s)
        d_error_da = error * y_i

        # Update parameters using gradient descent (mean of gradients over all x values)
        slopes[i] -= lr * np.mean(d_error_ds)
        shifts[i] -= lr * np.mean(d_error_dsh)
        amps[i]   -= lr * np.mean(d_error_da)

        # Update the visual data for each individual sigmoid line
        lines_sigmoids[i].set_data(x, amps[i] * sigmoid(slopes[i] * (x - shifts[i])))

    # Gradient for the global output bias
    # d(Error)/d(output_bias) = error * d(combined_output)/d(output_bias) = error * 1
    d_error_dbias = error
    output_bias -= lr * np.mean(d_error_dbias)

    # Update the visual data for the combined output line
    line_combined.set_data(x, combined_output)
    
    # Update title to show current frame and Mean Squared Error, indicating progress
    ax.set_title(f"Neural Network Fitting Complex Shape (Frame: {frame+1}/{25000}) - MSE: {mse:.6f}")

    return lines_sigmoids + [line_combined]

# Create the animation with significantly more frames for a better fit
# Increase frames to 25000 for more training time
ani = animation.FuncAnimation(fig, update, frames=25000, interval=20, blit=True, repeat=False)

plt.show()