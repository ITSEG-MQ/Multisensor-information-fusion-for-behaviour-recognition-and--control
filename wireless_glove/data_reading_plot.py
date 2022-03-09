import numpy as np
import matplotlib.pyplot as plt

channel = 8

root = "DEADLIFT/"
fitness_data = np.loadtxt('48.txt', delimiter=',')

fitness_data = np.array(fitness_data)

print(np.shape(fitness_data))

plt.figure()
for index in range(channel):
    plt.subplot2grid((8, 2), (index, 0), colspan=1, rowspan=1)
    plt.plot(fitness_data[index])

# acc plot
acc = plt.subplot2grid((8, 2), (0, 1), rowspan=4)
plt.plot(fitness_data[8], color='red')
plt.plot(fitness_data[9], color='green')
plt.plot(fitness_data[10], color='blue')
acc.set_title("Acceleration")
        
# gyro plot
gyro = plt.subplot2grid((8, 2), (4, 1), rowspan=4)
plt.plot(fitness_data[11], color='red')
plt.plot(fitness_data[12], color='green')
plt.plot(fitness_data[13], color='blue')
gyro.set_title("Gyroscope")
        
plt.tight_layout()
plt.show()