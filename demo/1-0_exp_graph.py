import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 400)
y = np.exp(x)

plt.figure(figsize=(7, 5))
plt.plot(x, y, color='#1f77b4', linewidth=2, label=r'$y = e^x$')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title(r'График $y = e^x$ на $[-1,\,1]$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.xlim(-2.1, 2.1)
plt.ylim(0, 3)
plt.tight_layout()
plt.show()
