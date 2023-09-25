import numpy as np
from compmec.nurbs import Curve
from matplotlib import pyplot as plt

allvectors = {}
allctrlpoints = {}

allvectors["N"] = (0, 0, 1, 2, 3, 3)
allctrlpoints["N"] = [(0, 0), (0, 3), (2, 0), (2, 3)]

allvectors["U0"] = (0, 0, 1, 1)
allctrlpoints["U0"] = [(6, 3), (6, 1)]
allvectors["U1"] = (1, 1, 1, 1.5, 2, 2, 2)
allctrlpoints["U1"] = [(6, 1), (6, 0), (8, 0), (8, 1)]
allvectors["U2"] = (2, 2, 3, 3)
allctrlpoints["U2"] = [(8, 1), (8, 3)]

allvectors["R0"] = [0, 0, 1, 2, 2]
allctrlpoints["R0"] = [(12, 0), (12, 3), (13, 3)]
allvectors["R1"] = [2, 2, 2, 3, 4, 4, 4]
allctrlpoints["R1"] = [(13, 3), (14, 3), (14, 1.5), (13, 1.5)]
allvectors["R2"] = [4, 4, 5, 5]
allctrlpoints["R2"] = [(13, 1.5), (12, 1.5)]
allvectors["R3"] = [5, 5, 6, 6]
allctrlpoints["R3"] = [(13, 1.5), (14, 0)]

allvectors["B0"] = [0, 0, 1, 2, 2]
allctrlpoints["B0"] = [(0, 0), (0, 3), (1, 3)]
allvectors["B1"] = [2, 2, 2, 3, 4, 4, 5, 6, 6, 6]
allctrlpoints["B1"] = [(1, 3), (2, 3), (2, 1.5), (1, 1.5), (2, 1.5), (2, 0), (1, 0)]
allvectors["B2"] = [4, 4, 5, 5]
allctrlpoints["B2"] = [(0, 1.5), (1, 1.5)]
allvectors["B3"] = [4, 4, 5, 5]
allctrlpoints["B3"] = [(0, 0), (1, 0)]

allvectors["S0"] = [0, 0, 0, 1, 2, 2, 3, 4, 4, 4]
allctrlpoints["S0"] = [(2, 3), (0, 3), (0, 1.5), (1, 1.5), (2, 1.5), (2, 0), (0, 0)]


allvectors["H"] = [0, 0, 0, 1, 2, 2, 2]
allctrlpoints["H"] = [(-9, 0), (-4.5, 1), (4.5, -1), (9, 0)]

curves = {}
for letter in allvectors:
    if letter not in allctrlpoints:
        continue
    points = np.array(allctrlpoints[letter])
    if letter.startswith("U"):
        points[:, 0] -= 2
    if letter.startswith("R"):
        points[:, 0] -= 4
    if letter.startswith("B"):
        points[:, 0] += 12

    if letter.startswith("S"):
        points[:, 0] += 16
    if letter.startswith("H"):
        points[:, 0] += 9
        points[:, 1] -= 1.5

    curve = Curve(allvectors[letter], points)
    curves[letter] = curve


fig = plt.figure(figsize=(9, 3))

for letter, curve in curves.items():
    curve.clean()
    umin, umax = curve.knotvector.limits
    uplot = np.linspace(umin, umax, 129)

    xvals, yvals = np.array(curve(uplot)).T
    plt.plot(xvals, yvals, color="deepskyblue", linewidth=5)

    xvals, yvals = np.array(curve.ctrlpoints).T
    plt.plot(xvals, yvals, linewidth=1, marker="o", ls="dotted", color="blue")


# xvals, yvals = np.array(curve.ctrlpoints).T
# plt.plot(xvals, yvals, color="r", ls="dotted", marker="o")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.axis("equal")
# plt.show()
plt.savefig("NURBS-logo.svg")
