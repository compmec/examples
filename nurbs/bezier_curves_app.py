# import matplotlib
# matplotlib.use('webagg')

import math
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
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


class WeigthedNodes:
    """ """

    def __init__(self):
        self.__degree = None
        self.__nodes = None
        self.__T = None

    def __compute_caract_matrix(self) -> Tuple[Tuple[float]]:
        """
        It's possible to find a caracteristic matrix [M] such

        C(u) = [P0, ..., Pp] * [M] * [1, u, u^2, ..., u^p]
        """
        if self.degree is None:
            return
        matrix = np.zeros((self.degree + 1, self.degree + 1), dtype="float64")
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                val = math.comb(self.degree, i) * math.comb(self.degree - i, j)
                val *= -1 if (self.degree + i + j) % 2 else 1
                matrix[i, j] = val
        matrix = matrix[:, ::-1]
        self.__M = matrix

    def __compute_transformation_matrix(self):
        """
        Computes the matrix [T] such

        C(u) = [P0, ..., Pp] * [T]
        C(u_j) = sum_{i=0}^{p} P_i * T_{ij}
        """
        if self.__degree is None or self.__nodes is None:
            return
        matrix = np.zeros((self.degree + 1, len(self.nodes)), dtype="float64")
        matrix[0, :] = 1
        for i in range(self.degree):
            matrix[i + 1] = self.nodes * matrix[i]
        self.__T = np.dot(self.__M, matrix)

    def eval(self, ctrlpoints: Tuple[Tuple[float]]):
        return np.dot(ctrlpoints, self.__T)

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def nodes(self) -> int:
        return self.__nodes

    @property
    def trans_matrix(self) -> Tuple[Tuple[float]]:
        return self.__T

    @degree.setter
    def degree(self, value: int):
        self.__degree = int(value)
        self.__compute_caract_matrix()
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

    def __init__(self, control_polygon, ax_bernstein):
        """Constructor.
        Receives the initial control polygon of the curve.
        """
        self.control_polygon = control_polygon
        xpts = tuple(control_polygon.get_xdata())
        ypts = tuple(control_polygon.get_ydata())
        self.points = [Point2D(x, y) for x, y in zip(xpts, ypts)]
        self.canvas = control_polygon.figure.canvas
        self.ax_main = control_polygon.axes
        self.ax_bernstein = ax_bernstein
        self.mouse = Mouse()
        self.__max_norm2 = 0

        # Event handler for mouse clicking
        cid = self.canvas.mpl_connect("button_press_event", self.__mouse_press)
        self.cid_button_press = cid
        cid = self.canvas.mpl_connect("button_release_event", self.__mouse_release)
        self.cid_button_press = cid
        cid = self.canvas.mpl_connect("motion_notify_event", self.__mouse_move)
        self.cid_button_press = cid

        self.__weighted_nodes = WeigthedNodes()
        self.__weighted_nodes.nodes = np.linspace(0, 1, 1029)

        # Create Bézier curve
        line_bezier = Line2D([], [], c=control_polygon.get_markeredgecolor())
        self.bezier_curve = self.ax_main.add_line(line_bezier)

    def __find_point(self, x: float, y: float) -> Union[int, None]:
        index = 0
        while index < len(self.points):
            point = self.points[index]
            norm2 = (point[0] - x) ** 2 + (point[1] - y) ** 2
            if norm2 < 4e-4 * self.__max_norm2:  # tolerance
                return index
            index += 1
        return None

    def __mouse_press(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.control_polygon.axes:
            return
        x, y = event.xdata, event.ydata
        if event.button == 3:  # RIGHT click
            self.__remove_point(x, y)
            return
        if event.dblclick:
            self.__add_point(x, y)
            return
        self.mouse.clicked = True
        self.mouse[0] = x
        self.mouse[1] = y
        self.__index_point = self.__find_point(x, y)

    def __mouse_release(self, event):
        self.mouse.clicked = False
        self.__index_point = None

    def __mouse_move(self, event):
        if event.inaxes != self.control_polygon.axes:
            return
        x, y = event.xdata, event.ydata
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

    def __move_point(self, index: int, dx: float, dy: float):
        self.points[index].move(dx, dy)
        self.__update_all()

    def __add_point(self, x: float, y: float):
        # Add point
        new_point = Point2D(x, y)
        self.points.append(new_point)
        self.__weighted_nodes.degree = len(self.points) - 1
        self.__max_norm2 = max(point.norm2() for point in self.points)
        self.__update_all()

    def __remove_point(self, x: float, y: float):
        # Remove point
        index = self.__find_point(x, y)
        if index is not None:
            self.points.pop(index)
            self.__weighted_nodes.degree = len(self.points) - 1
            self.__update_all()

    def __update_all(self):
        xpts = tuple(point[0] for point in self.points)
        ypts = tuple(point[1] for point in self.points)
        self.control_polygon.set_data(xpts, ypts)
        # Rebuild Bézier curve and update canvas
        xvals = self.__weighted_nodes.eval(xpts)
        yvals = self.__weighted_nodes.eval(ypts)
        self.bezier_curve.set_data(xvals, yvals)
        self._update_bernstein()
        self._update_bezier()

    def _update_bezier(self):
        self.canvas.draw()

    def _update_bernstein(self):
        nodes = self.__weighted_nodes.nodes
        matrix = self.__weighted_nodes.trans_matrix
        ax = self.ax_bernstein
        ax.clear()
        for i, line in enumerate(matrix):
            ax.plot(nodes, line)
        ax.set_title("Bernstein basis, N = {}".format(matrix.shape[0]))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


if __name__ == "__main__":
    # Initial setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Empty line
    line = Line2D([], [], ls="--", c="#666666", marker="x", mew=2, mec="#204a87")
    ax1.add_line(line)

    # Canvas limits
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Bézier curve")

    # Bernstein plot
    ax2.set_title("Bernstein basis")

    # Create BezierBuilder
    bezier_builder = BezierBuilder(line, ax2)

    plt.show()
