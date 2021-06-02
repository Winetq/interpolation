from functions import *


def main():
    files = ['MountEverest.csv', 'SpacerniakGdansk.csv', 'WielkiKanionKolorado.csv']
    for i in range(len(files)):
        x, y, n = load_data(files[i])
        x1, y1 = load_data_with_n(files[i], 22)
        x1_Lagrange, y1_Lagrange = Lagrange_algorithm(x, x1, y1)
        count_RMSD(y, y1_Lagrange)
        comparison(x1, y1, x, y, x1_Lagrange, y1_Lagrange, 'interpolacja metodą Lagrange')
        x1_spline, y1_spline = spline_algorithm(x, x1, y1)
        count_RMSD(y, y1_spline)
        comparison(x1, y1, x, y, x1_spline, y1_spline, 'interpolacja metodą splajnów 3-stopnia')


main()

