import csv
import matplotlib.pyplot as plt
import numpy as np

# Extract data from csv file
file = open(r"C:\Users\20167271\Desktop\ML for signal processing\A1\assignment1_data.csv")
csvreader = csv.reader(file)
header = ['x[k]', 'y[k]']  # next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
file.close()


# Initialize filter
alpha = 0.145 #np.random.random(1)  # ensures convergence
N = len(rows)  # number of samples
D = 3 #len(rows[0][0])  # input dimension
w0 = np.random.rand(D,1)  # initialize weights from random distribution
ryx = np.array([[1],[5.3],[-3.9]])
Rx = np.array([[5, -1, -2], [-1, 5, -1], [-2, -1, 5]])
RxInv = np.linalg.inv(Rx)
print("Alpha was chosen as: ", alpha)


# Implement algorithm
whist = [w0.tolist()]
for i in range(N):
    w1 = w0 + 2*alpha*np.matmul(RxInv, ryx - np.matmul(Rx, w0))
    whist.append(w1.tolist())
    w0 = w1


# Plot
whist = np.array(whist)
p = plt.plot(whist[:,0], whist[:,1], '-o')
plt.plot(whist[0,0], whist[0,1], 'r-o', markeredgewidth = 2)
plt.plot(whist[1001,0], whist[1001,1], 'g-o', markeredgewidth = 2)
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Weight convergence with Newton")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex123i.png")
plt.show()