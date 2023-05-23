import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4
gradient = lambda x: 2*x - 4

x = -10.0
epochs = 20
learning_rate = 0.25

x_values = np.linspace(-10, 10, 100)
y_values = f(x_values)

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    print(i + 1, x, f(x))

plt.plot(x_values, y_values, 'k-')
plt.plot(x, f(x), 'ro')  # Plot the solution point
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#그림