# import matplotlib
# matplotlib.use('webagg')

from __future__ import annotations

import math
import sys
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from compmec import nurbs
from matplotlib.lines import Line2D

"""
A bezier curve is defined as

C(u) = sum_{i=0}^{p} B_{i,p}(u) * P_{i,p}
B_{i,p}(u) = binom(p, i) * (1-u)^{p-i} * u^i

"""


class Point2D:
    def __init__(self, x: float, y: float):
        self.update(x, y)

    def __str__(self) -> str:
        return f"({self[0]}, {self[1]})"

    def __getitem__(self, index: int) -> float:
        return self.__x if index == 0 else self.__y

    def move(self, dx: float, dy: float):
        self.__x += dx
        self.__y += dy

    def update(self, x: float, y: float):
        self.__x = x
        self.__y = y

    def norm2(self) -> float:
        return self.__x**2 + self.__y**2

    def inner(self, other: Point2D) -> float:
        return self[0] * other[0] + self[1] * other[1]

    def __mul__(self, other: Point2D) -> float:
        return self.inner(other)

    def __rmul__(self, value: float) -> Point2D:
        return self.__class__(value * self[0], value * self[1])

    def __add__(self, other: Point2D) -> Point2D:
        return self.__class__(self[0] + other[0], self[1] + other[1])


class WeigthedNodes:
    """ """

    def __init__(self):
        self.__basis = None
        self.__nodes = None
        self.__T = None

    def __compute_transformation_matrix(self):
        if self.__basis is None or self.__nodes is None:
            return
        self.__T = np.array(self.__basis.eval(self.nodes), dtype="float64")

    def eval(self, ctrlpoints: Tuple[Tuple[float]]):
        return np.dot(ctrlpoints, self.__T)

    @property
    def knotvector(self) -> Tuple[float]:
        return self.__basis.knotvector

    @property
    def nodes(self) -> int:
        return self.__nodes

    @property
    def trans_matrix(self) -> Tuple[Tuple[float]]:
        return self.__T

    @knotvector.setter
    def knotvector(self, vector: Tuple[float]):
        self.__basis = nurbs.Function(vector)
        self.__compute_transformation_matrix()

    @nodes.setter
    def nodes(self, values: Tuple[float]):
        self.__nodes = np.array(values, dtype="float64")
        self.__compute_transformation_matrix()


class Mouse:
    def __init__(self):
        self.__x = None
        self.__y = None
        self.clicked = False

    def __getitem__(self, index: int) -> float:
        return self.__x if index == 0 else self.__y

    def __setitem__(self, index: int, value: float):
        if index == 0:
            self.__x = value
        else:
            self.__y = value


class BezierBuilder(object):
    """Bézier curve interactive builder."""

    def __init__(self, control_polygon, ax_bspline):
        """Constructor.
        Receives the initial control polygon of the curve.
        """
        self.control_polygon = control_polygon
        self.canvas = control_polygon.figure.canvas
        self.ax_main = control_polygon.axes
        self.ax_bspline = ax_bspline
        self.mouse = Mouse()
        self.__index_point = None

        # Event handler for mouse clicking
        cid = self.canvas.mpl_connect("button_press_event", self.__mouse_press)
        self.cid_mouse_press = cid
        cid = self.canvas.mpl_connect("button_release_event", self.__mouse_release)
        self.cid_mouse_release = cid
        cid = self.canvas.mpl_connect("motion_notify_event", self.__mouse_move)
        self.cid_mouse_move = cid
        cid = self.canvas.mpl_connect("key_press_event", self.__key_press)
        self.cid_key_press = cid

        xpts = tuple(control_polygon.get_xdata())
        ypts = tuple(control_polygon.get_ydata())
        max_norm2 = 0
        for i, (xi, yi) in enumerate(zip(xpts, ypts)):
            for j in range(i + 1, len(xpts)):
                max_norm2 = max(max_norm2, (xi - xpts[j]) ** 2 + (yi - ypts[j]) ** 2)
        self.__max_norm2 = max_norm2

        knotvector = nurbs.GeneratorKnotVector.bezier(len(xpts) - 1)
        self.weighted_nodes = WeigthedNodes()
        self.weighted_nodes.knotvector = knotvector
        self.weighted_nodes.nodes = np.linspace(0, 1, 1029)
        self.points = [Point2D(x, y) for x, y in zip(xpts, ypts)]

        # Create Bézier curve
        line_bezier = Line2D(
            [0.2, 0.8], [0.2, 0.8], c=control_polygon.get_markeredgecolor()
        )
        self.bezier_curve = self.ax_main.add_line(line_bezier)

        self.__update_all()

    def __find_point(self, x: float, y: float) -> Union[int, None]:
        tolerance = 4e-2 * self.__max_norm2
        norm2s = [(point[0] - x) ** 2 + (point[1] - y) ** 2 for point in self.points]
        indexs = [i for i, norm2 in enumerate(norm2s) if norm2 < tolerance]
        if len(indexs) == 0:
            return None
        minnorm2 = min(norm2s[i] for i in indexs)
        return norm2s.index(minnorm2)

    def __mouse_press(self, event):
        # Ignore clicks outside axes
        x, y = event.xdata, event.ydata
        if event.inaxes == self.control_polygon.axes:
            self.mouse.clicked = True
            self.mouse[0] = x
            self.mouse[1] = y
            self.__index_point = self.__find_point(x, y)
        elif event.inaxes == self.ax_bspline:
            if event.button == 1:  # LEFT click
                self.__knot_insert(x)
            if event.button == 3:  # RIGHT click
                knotvector = self.weighted_nodes.knotvector
                mult = knotvector.mult(x)
                if mult == 0:
                    for knot in knotvector:
                        if abs(knot - x) < 3e-3:
                            x = knot
                            break
                self.__knot_remove(x)

    def __mouse_release(self, event):
        self.mouse.clicked = False
        self.__index_point = None

    def __mouse_move(self, event):
        x, y = event.xdata, event.ydata
        if event.inaxes == self.control_polygon.axes:
            if not self.mouse.clicked or self.__index_point is None:
                self.mouse[0] = x
                self.mouse[1] = y
                return

            dx = x - self.mouse[0]
            dy = y - self.mouse[1]
            self.mouse[0] = x
            self.mouse[1] = y
            index = self.__index_point
            self.__move_point(index, dx, dy)
        elif event.inaxes == self.ax_bspline:
            self.ax_bspline.axvline(x=x)
            self._update_bspline()

    def __key_press(self, event):
        if event.key == "+":
            self.__increase_degree()
        elif event.key == "-":
            self.__decrease_degree()

    def __move_point(self, index: int, dx: float, dy: float):
        self.points[index].move(dx, dy)
        self.__update_all()

    def __increase_degree(self):
        knotvector = self.weighted_nodes.knotvector
        curve = nurbs.Curve(knotvector, self.points)
        curve.degree_increase()
        self.weighted_nodes.knotvector = curve.knotvector
        self.points = curve.ctrlpoints
        self.__update_all()

    def __decrease_degree(self):
        try:
            knotvector = self.weighted_nodes.knotvector
            curve = nurbs.Curve(knotvector, self.points)
            curve.degree_decrease(tolerance=None)
            self.weighted_nodes.knotvector = curve.knotvector
            self.points = curve.ctrlpoints
            self.__update_all()
        except ValueError:
            pass

    def __knot_insert(self, knot: float):
        try:
            knotvector = self.weighted_nodes.knotvector
            curve = nurbs.Curve(knotvector, self.points)
            curve.knot_insert([knot])
            self.weighted_nodes.knotvector = curve.knotvector
            new_nodes = sorted(list(self.weighted_nodes.nodes) + [knot])
            self.weighted_nodes.nodes = new_nodes
            self.points = curve.ctrlpoints
            self.__update_all()
        except ValueError:
            pass

    def __knot_remove(self, knot: float):
        try:
            knotvector = self.weighted_nodes.knotvector
            curve = nurbs.Curve(knotvector, self.points)
            curve.knot_remove([knot], tolerance=None)
            self.weighted_nodes.knotvector = curve.knotvector
            self.points = curve.ctrlpoints
            self.__update_all()
        except ValueError:
            pass

    def __update_all(self):
        xpts = tuple(point[0] for point in self.points)
        ypts = tuple(point[1] for point in self.points)
        self.control_polygon.set_data(xpts, ypts)
        # Rebuild Bézier curve and update canvas
        xvals = self.weighted_nodes.eval(xpts)
        yvals = self.weighted_nodes.eval(ypts)
        self.bezier_curve.set_data(xvals, yvals)
        self._update_bspline()
        self._update_bezier()

    def _update_bezier(self):
        self.canvas.draw()

    def _update_bspline(self):
        nodes = self.weighted_nodes.nodes
        knotvector = self.weighted_nodes.knotvector
        matrix = self.weighted_nodes.trans_matrix
        ax = self.ax_bspline
        ax.clear()
        for i, line in enumerate(matrix):
            ax.plot(nodes, line)
        for knot in knotvector.knots[1:-1]:
            ax.axvline(x=knot, color="k", ls="dotted")
        title = f"Bspline basis of degree {knotvector.degree} "
        title += f"and {knotvector.npts} control points"
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


if __name__ == "__main__":
    # Initial setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Empty line
    line = Line2D(
        [0.2, 0.8], [0.2, 0.8], ls="--", c="#666666", marker="x", mew=2, mec="#204a87"
    )
    ax1.add_line(line)

    # Canvas limits
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Bézier curve")

    # bspline plot
    ax2.set_title("bspline basis")

    # Create BezierBuilder
    bezier_builder = BezierBuilder(line, ax2)

    plt.show()
