import matplotlib.pyplot as plt
import math
e = math.e
x = [0,1,2,3]
y= [1,100,10000,1000000]

#plot by log scale
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Actual Price', linestyle='-', marker='o')
plt.yscale('log')  # Logarithmic scale
plt.show()