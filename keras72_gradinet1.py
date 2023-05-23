import numpy as np
import matplotlib.pyplot as plt

f = lambda x : 2*x - 4 + 6
x = np.linspace(-1,6,100)
print(x,len(x))

y = f(x)

plt.plot(x,y,'k-')
plt.plot(2,2,'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'k-')