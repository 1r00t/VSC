import numpy as np


# (a) Berechnen Sie den Winkel $\alpha$ in Grad zwischen den folgenden beiden Vektoren $a=[1.,1.77]$ und $b=[1.5,1.5]$
a = np.array([-1., 1.77])
b = np.array([1.5, 1.5])
# YOUR CODE HERE

a_len = np.linalg.norm(a)
b_len = np.linalg.norm(b)
# c = math.acos((a[0] * b[0] + a[1] + b[1]) / (math.sqrt(a[0]*a[0] + a[1] * a[1]) * math.sqrt(b[0]*b[0] + b[1] * b[1])))
print(np.degrees(np.arccos(a.dot(b) / (a_len * b_len))))


# (b) Gegeben ist die quadratische regulaere Matrix A und ein Ergbnisvektor b. Rechnen Sie unter Nutzung der Inversen die Loesung x des Gleichungssystems Ax = b aus.
# YOUR CODE HERE
A = np.matrix('2 3 4; 3 -2 -1; 5 4 3')
b = np.matrix('1.4; 1.2; 1.4')
print(np.linalg.inv(A) * b)


# (c) Schreiben Sie eine Funktion die das Matrixprodukt berechnet. Nutzen Sie dafür nicht die Numpy Implementierung.
# Hinweis: Fangen Sie bitte mögliche falsche Eingabegroessen in der Funktion ab und werfen einen AssertionError
# assert Expression[, Arguments]

def matmult(M1, M2):
    assert(M1.shape[1] == M2.shape[0])
    result = np.zeros((M1.shape[0], M2.shape[1]))
    for i in range(M1.shape[0]):
        for j in range(M2.shape[1]):
            for k in range(M1.shape[1]):
                result[i, j] += M1[i, k] * M2[k, j]
    return result


M1 = np.matrix('1 2; 3 4; 5 6')
M2 = np.matrix('2 0; 0 2')

M_res = matmult(M1, M2)
print(M_res)
