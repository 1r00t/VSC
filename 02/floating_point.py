import numpy as np
from math import sqrt, fabs, pi
import matplotlib.pyplot as plt

# Aufgabe 1 (a)
# i ist hier die Anzahl der Iterationen
# In jeder Iteration soll ein epsilon auf 1.0 addiert werden und mit der
# Floating-Point Darstellung von np.float64(1) bzw. np.float(32) verglichen werden.
# Starten Sie dabei mit Epsilon=1.0 und halbieren Sie den Wert in jeder Iteration (wie an der Ausgabe 2^(-i) zu sehen)
# Stoppen Sie die Iterationen, wenn np.float32(1) + epsi != np.float32(1) ist.
# Hinweis: Ja - in diesem Fall dürfen Sie Floating-Point Werte vergleichen ;)

epsi64 = np.float64(1)
epsi32 = np.float32(1)
i = 0

# Print Anweisung vor dem Loop
print('i | 2^(-i) | 1 + 2^(-i) ')
print('----------------------------------------')


while np.float64(1) + epsi64 != np.float64(1):
    epsi64 = epsi64 / np.float64(2)
    i += 1
# Print Anweisung in / nach dem Loop
print('{0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi64))

i = 0
while np.float32(1) + epsi32 != np.float32(1):
    epsi32 = epsi32 / np.float32(2)
    i += 1
print('{0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi32))


# exit(0)

# Aufgabe 1 (b)
# Werten Sie 30 Iterationen aus und speichern Sie den Fehler in einem
# Fehlerarray err
N = 30
err = []
# sqrt(2) kann vorberechnet werden
sn = sqrt(2)

for n in range(2, N):
    # 1. Umfang u berechnen
    # 2. Fehler en berechnen und in err speichern
    # Fehler ausgeben print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))
    # YOUR CODE HERE
    en = abs(2 * pi - sn * 2 ** n)
    err.append(en)
    u = sqrt(2 - sqrt(4 - sn * sn))
    sn = u
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))

# Plotten Sie den Fehler
plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()


# Aufgabe 1 (c)
# Löschen des Arrays und wir fangen mit der Berechnung von vorn an.
# Nur diesmal mit der leicht veranderten Variante
err = []
sn = sqrt(2)

for n in range(2, N):
    en = abs(2 * pi - sn * 2 ** n)
    err.append(en)
    u = sn / sqrt(2 + sqrt(4 - sn * sn))
    sn = u
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))


plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()
