import numpy as np
import matplotlib.pyplot as plt
import tomograph


def show_phantom(size):
    """
    Hilfsfunktion um sich das Originalbild anzuschauen.
    """
    I = tomograph.phantom(size)
    plt.imshow(I, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0], origin='lower', interpolation='nearest')
    plt.show()

# show_phantom(128)


def create_sinogram(angle, nAngles, nSamples):
    """
    Funktion soll Sinogram erzeugen

    :param angle: Winkel ueber die die Strahlen laufen (0-180 Grad) in rad
    :param nAngles: Anzahl der Winkelschritte bis angle
    :param nSamples: Anzahl der Strahlen pro Winkel

    :return: Tuple sinogram matrix, Strahlstartpunkt, Strahlrichtung
    """
    # YOU CODE HERE
    # Anlegen von leeren Matrizen fuer Strahlstart und -richtung
    # rp - Strahlstart: pro Gitterpunkt i,j eine x-y-Position
    # rd - Strahlrichtung: pro Winkelschritt ein x-y-Richtung
    rp = np.zeros((nAngles, nSamples, 2))
    rd = np.zeros((nAngles, 2))

    # ueber all Winkel laufen
    for j, a in enumerate(np.linspace(0, angle, nAngles)):
        # Winkel hier in Polarkoordinatendarstellung x/y Position
        x = np.cos(a)
        y = np.sin(a)
        rd[j] = np.array([-x, -y])

    # auf dem Einheitskreis um das Phantom - das ergibt dann die Strahlrichtung

    # an jedem Strahlursprung nach links und rechts gehen (auf einer Geraden) und
    # Strahlstartwerte berechnen, TIP: die vorher berechneten x/y Positionen
    # ergeben als vektor eine Strahlrichtung rd[j] -> np.array([-x, -y])

    # Die Strahlstartpositionen der Strahlenfront ergeben sich ueber
    # wobei sich g durch den loop
        for i, g in enumerate(np.linspace(-0.99, 0.99, nSamples)):
            rp[i, j] = np.array([x - y * g, y + x * g])

    # Ein sinogramm ist dann ein Array mit abgeschwaechten Intensitaeten pro Winkel
    # und Strahl, d.h. die Matrix ist Anzahl Strahlen x Anzahl der Winkel, bzw.
    # Anzahl der Strahlen pro Aufnahme und die Anzahl der Aufnahmen.
    sinogram = np.zeros((nSamples, nAngles))

    for j in range(nAngles):
        # trace-Funktion aufrufen und sinogram-Matrix fuellen
        for i in range(nSamples):
            sinogram[i, j] = tomograph.trace(rp[i, j], rd[j])

    return sinogram, rp, rd


# plot mit unterfigures
plt.figure(figsize=(16, 8))
sf = 1

# nur für Grid >1 erlaubt
gridsizes = [32, 64]  # , 64, 128, 256]
for ng in gridsizes:
    print("GRID: ", ng)
    nGrid = ng
    # die Anzahl der Winkelstufen
    nSamples = 2 * nGrid
    nAngles = 2 * nGrid

    sinogram, rp, rd = create_sinogram(np.pi, nAngles, nSamples)
    # plt.show()
    # Die bekannte aufgenommenen Intensitaetswerte im Sinogram speichern wir als ein Vektor (siehe np.ravel)
    # in logarithmierter Form ab also log ( I_0 / I_1)
    I_0 = 1.0
    I_1 = np.log(I_0 / np.ravel(sinogram))

    # Initialisieren Sie eine Matrix L in der gewuenschten Größe
    L = np.zeros((nSamples * nAngles, nGrid ** 2))

    # Für jeden Winkel rekonstruieren wir jetzt die Daten
    # Dafuer muessen Sie ueber alle Winkel die Funktion grid_intersect (Rückgabe -> I, G, dt)
    # nutzen. Die errechneten Strahllangen pro Quadrant (Pixel) sind dann die Eintraege
    # in die Matrix L. Da das etwas Indexmagic ist - hier der Zugriff:
    # L[I * nAngles + j, G] = ...
    for j in range(nAngles):
        # Parameter:
        # n  : Höhe bzw. Breite des für die Rekonstruktion verwendeten Rasters
        # r  : Startpunkte der Strahlen; gespeichert als Zeilen einer Matrix (ndarray)
        # d  : Richtungsvektor der Strahlen (array_like)
        I, G, dt = tomograph.grid_intersect(nGrid, rp[:, j], rd[j])
        L[I * nAngles + j, G] = dt

    # Loesen des Ausgleichsproblems mit Hilfe von
    X = np.linalg.lstsq(L, I_1)[0]

    # Loesungsvektor wieder auf die gewuenschte Groesse bringen - reshape

    X = X.reshape((nGrid, nGrid))

    # Plotten Sie das Sinogram mit Hilfe von Matplotlib. Nutzen Sie die 'gist_yarg' color map
    plt.subplot(2, 4, sf)
    plt.imshow(sinogram, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0], origin='lower', interpolation='nearest')

    # Plotten Sie die Rekonstruktion mit Hilfe von Matplotlib. Nutzen Sie die 'gist_yarg' color map
    plt.subplot(2, 4, sf + 4)
    plt.imshow(X, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0], origin='lower', interpolation='nearest')

    sf += 1
    plt.savefig('tg_fig2.pdf', bbox_inches='tight')
plt.show()
