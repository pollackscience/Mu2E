#!/usr/bin/env python

print("This is exercise_1.")

import numpy as np
import matplotlib.pyplot as plt

print("pi is " + str(np.pi))
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)

# Print out some values
for it in range(5):
    print("\ti= " + str(it) + "\tX= " + str(X[it]))

deltaX = X[1]-X[0]
print("step size = " + str(deltaX))

print("min and max:\t" + str(C.min()) + "\t" + str(C.max()))


# Make plots of curves

plt.figure(figsize=(8,6), dpi=80)

plt.subplot(111)

plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=2.0, linestyle="--")

plt.xlim(-4,4)
plt.ylim(-1.2,1.2)

plt.xticks(np.linspace(-4,4,9,endpoint=True))
plt.yticks(np.linspace(-1.2,1.2,13,endpoint=True))

#savefig("exercize_1.pdf",dpi=72)

plt.show()

# Comment
print("How does it look?")
