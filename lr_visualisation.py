import matplotlib.pyplot as plt
import numpy as np

N=360
y = np.logspace(0, -3, N, endpoint=True)
x = list(range(N))
x = np.array(x)

plt.plot(x, y, 'o')
plt.show()