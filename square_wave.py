import numpy as np
import matplotlib.pyplot as plt

def generate_square_wave(t: np.ndarray) -> np.ndarray:
    return np.piecewise(t, [t < -0.5, (t >= -0.5) & (t < 0.5), t >= 0.5], [-1, 1, 0])

# Define the time array
# This will create an array of values from -1.5 to 1.5, which is the domain of your function.
#t = np.linspace(-1.5, 1.5, 1000)  # 1000 points between -1.5 and 1.5
#
## Generate the square wave values for the adjusted function
#square_wave_custom_values:np.ndarray = generate_square_wave(t)
#
## Plotting the adjusted square wave
#plt.figure(figsize=(6, 3))
#plt.plot(t, square_wave_custom_values, 'b-', linewidth=2)
#plt.xlim([-1, 1])
#plt.ylim([-1.5, 1.5])
#plt.grid(True)
#plt.title('Custom Square Wave Graph')
#plt.show()

