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
lambda_max = 1/70
alpha = 0.0004 #np.random.randn(1) * lambda_max  # ensures convergence
N = len(rows)  # number of samples
D = 3 #len(rows[0][0])  # input dimension
w0 = np.zeros((D,1))  
print("Choice of alpha is: ",alpha)


# Implement algorithm
whist = [w0.tolist()]
J = []
for i in range(N-2):
    x = xs[i:i+3].reshape((3,1))
    w1 = w0 + 2*alpha*x @ (ys[i+2] - np.transpose(x)@w0)
    whist.append(w1.tolist())
    w0 = w1


# w0 against w1
whist = np.array(whist)
p = plt.plot(whist[:,0], whist[:,1], '-o')
plt.plot(whist[0,0], whist[0,1], 'r-o', markeredgewidth = 2)
plt.plot(whist[1001,0], whist[1001,1], 'g-o', markeredgewidth = 2)
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Weight convergence with LMS")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex131j.png")
plt.show()


# Implement NORMALIZED algorithm
alpha = 0.0001 #np.random.randn(1) * lambda_max  # ensures convergence

whist = [w0.tolist()]
J = []
for i in range(N-2):
    x = xs[i:i+3].reshape((3,1))
    xsofar = xs[0:i+3]
    Ex = 1/len(xsofar) * np.transpose(xsofar) @ xsofar
    w1 = w0 + 2/Ex*alpha*x @ (ys[i+2] - np.transpose(x)@w0)
    whist.append(w1.tolist())
    w0 = w1


# w0 against w1
whist = np.array(whist)
p = plt.plot(whist[:,0], whist[:,1], '-o')
plt.plot(whist[0,0], whist[0,1], 'r-o', markeredgewidth = 2)
plt.plot(whist[1001,0], whist[1001,1], 'g-o', markeredgewidth = 2)
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Weight convergence with normalized LMS")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex131k.png")
plt.show()
