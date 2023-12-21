from square_wave import generate_square_wave
import matplotlib.pyplot as plt
import numpy as np
import torch

# Define the time array
t = np.linspace(-1.5, 1.5, 100_000)  # 1000 points between -1.5 and 1.5

square_wave_values: np.ndarray = generate_square_wave(t)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(1, 5)  # Input layer with 1 feature and 10 neurons
        self.fc2 = torch.nn.Linear(5, 5)  # Hidden layer with 10 neurons
        self.fc3 = torch.nn.Linear(5, 1)  # Output layer with 1 neuron (output)

    def forward(self, x):
        # Forward pass through the network
        x = torch.nn.functional.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.nn.functional.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)          # Output layer (no activation for regression)
        return x


model = Model()

# reshape input data and labels
xs = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)  # Reshaping to (1000, 1)
ys = torch.tensor(square_wave_values.reshape(-1, 1), dtype=torch.float32)

# use Mean Squared Error (MSE) as the loss function
criterion = torch.nn.MSELoss()

# use adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    # zero the gradients
    optimizer.zero_grad()

    #forward pass
    y_pred = model(xs)

    #compute loss
    loss = criterion(y_pred, ys)
    if epoch % 100 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item()}')

    #backward pass
    loss.backward()

    #update weights
    optimizer.step()

# After training, run ys through the trained network and plot the results
predicted_after_training = model(xs).detach().numpy()

# Plot the results after training
plt.figure(figsize=(10, 5))
plt.plot(t, predicted_after_training, label='Predicted by Trained Model', alpha=0.5)
plt.plot(t, square_wave_values, label='Original Square Wave', alpha=0.5)
plt.legend()
plt.title("Comparison of Trained Model Prediction and Original Square Wave")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.show()
