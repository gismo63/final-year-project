import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#MAE = np.array([0.063,0.038,0.024,0.016,0.011, 0.0086, 0.0068])
#x = np.arange(2,9)

MAE = np.array([0.078,0.035,0.0265, 0.019, 0.016, 0.015 ])
x = np.array([100,1000,4000, 10000, 25000, 50000])

plt.figure()
plt.plot(x, MAE)
plt.xlabel('Sample Size')
plt.ylabel('Mean Absolute Error')
plt.title("Error Scaling for Varying Sample Size: n = 5")
plt.grid()
plt.axhline(y=0, color='k')
plt.savefig('5N_sample.eps', format='eps', dpi=1200)
plt.show()
