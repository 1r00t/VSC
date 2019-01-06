import numpy as np
import matplotlib.pyplot as plt

# Laden der gegebenen Daten d0 - d4

x = np.linspace(-2, 2, 200)
d = []
for i in range(5):
    d.append(np.load(f'data/d{i}.npy'))
    plt.subplot(2, 3, i + 1)
    plt.plot(x, d[i], '.')
plt.show()


# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.


def create_matrix(x, degree):
    # YOUR CODE HERE
    A = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        A[:, i] = x ** i
    return np.flip(A, 1)


for i in range(1, 2):
    A = create_matrix(x, i)
    xx = np.linalg.lstsq(A, d[0], rcond=None)[0]  # ich weiß
    res = (d[0] - A.dot(xx)).T * (d[0] - A.dot(xx))
    print(res)

    # Krankheit / Faulheit / Dummheit

# Lösen Sie das lineare Ausgleichsproblem
# Hinweis: Nutzen Sie bitte hier nicht np.linalg.lstsq!, sondern implementieren sie A^T A x = A^T b selbst

# Stellen Sie die Funktion mit Hilfe der ermittelten Koeffizienten mit matplotlib
# np.poly1d
