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
rows = np.array(rows, dtype=np.float64)
xs = rows[:,0]
ys = rows[:,1]

# Initialize filter
lambda_max = 1/7
alpha = 0.0045 #np.random.randn(1) * lambda_max  # ensures convergence
N = len(rows)  # number of samples
D = 3 #len(rows[0][0])  # input dimension
w0 = np.random.randn(D,1)  # initialize weights from random distribution
ryx = np.array([[1],[5.3],[-3.9]])
Rx = np.array([[5, -1, -2], [-1, 5, -1], [-2, -1, 5]])
print("Choice of alpha is: ",alpha)


# Implement algorithm
whist = [w0.tolist()]
J = []
for i in range(N):
    w1 = w0 + 2*alpha*(ryx - np.matmul(Rx, w0))
    whist.append(w1.tolist())
    w0 = w1


# w0 against w1
whist = np.array(whist)
p = plt.plot(whist[:,0], whist[:,1], '-o')
plt.plot(whist[0,0], whist[0,1], 'r-o', markeredgewidth = 2)
plt.plot(whist[1001,0], whist[1001,1], 'g-o', markeredgewidth = 2)
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Weight convergence with GD")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex122f.png")
plt.show()


# Calculate J 
Datapoints = 1000

w0 = np.linspace(-10, 10, Datapoints)
w1 = np.linspace(-10, 10, Datapoints)
w2 = -0.5
ryxt = np.transpose(ryx)

J = np.ones((Datapoints, Datapoints))

ExpY = np.transpose(ys)@ys  # expectatio of y
bigX = np.ones((N-2,D))
for j in range(N-2):        # make X matrix from the given array
    bigX[j,:] = xs[j:j+3]

for i in range(Datapoints):
    for j in range(Datapoints):
        w = np.array([[w0[i]], [w1[j]], [w2]])  # column vector
        wt = np.transpose(w)
        J[i,j] = ExpY + wt @ Rx @ w - wt @ ryx - ryxt @ w


# Contour J
plt.contourf(w0, w1, J)
plt.plot(1/5, 1, 'rx')
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Countour plot for function J")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex122fcontour.png")
plt.show()
