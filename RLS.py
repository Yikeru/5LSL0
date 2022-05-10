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
N = len(rows)  # number of samples
#xs = [rows[r][0] for r in range(N)]
#ys = [rows[r][1] for r in range(N)]
rows = np.array(rows, dtype=np.float64)
xs = rows[:,0]
ys = rows[:,1]


# Initialize filter
gamma = 1-(1e-3)
ryx = np.array([[0],[0],[0]])
Rxinv = np.identity(3)*1e-4
RxInvList = [Rxinv]
ryxList = [ryx]
wo = []

# Implement algorithm
for i in range(N-3):
    x = xs[i:i+3].reshape((3,1))
    y = ys[i+2]
    g = (Rxinv @ x) / (gamma**2 + (np.transpose(x) @ (Rxinv @ x)))
    Rxinv = gamma**-2 * (Rxinv - g @ (np.transpose(x) @ Rxinv))
    RxInvList.append(Rxinv)
    ryx = gamma**2 * ryx + x * y  # ???????????????????????????????Is this the right y or should it be y[i]
    ryxList.append(ryx)
    w = Rxinv @ ryx
    wo.append(w)


# Plot
whist = np.array(wo)
p = plt.plot(whist[:,0], whist[:,1], '-o')
plt.plot(whist[0,0], whist[0,1], 'r-o', markeredgewidth = 2)
plt.plot(whist[1001,0], whist[1001,1], 'g-o', markeredgewidth = 2)
plt.xlabel('$w_{0}$')
plt.ylabel("$w_{1}$")
plt.title("Weight convergence with Newton")
plt.savefig(r"C:\Users\20167271\Desktop\ML for signal processing\A1\ex132m.png")
plt.show()
