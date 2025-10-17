from numpy import sin, cos, pi, exp, real, imag
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.signal as sig
from ipywidgets import IntSlider, HBox, Layout, Output
import ipywidgets as widget
from IPython.display import display


class InteractiveDemo:
    def __init__(self, fig_num: int=None, figsize: tuple = (8,6)):
        self.layout = None
        if fig_num is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            plt.close(fig_num)
            self.fig = plt.figure(num=fig_num, figsize=figsize)
        
        self.init_plot()
        self.create_widgets()

        if self.layout is not None:
            out = Output()
            display(self.layout, out)
    
    def init_plot(self):
        raise NotImplementedError("Subclasses should implement 'init_plot' method.")
        
    def create_widgets(self):
        raise NotImplementedError("Subclasses should implement 'create_widgets' method.")



class FixedPointDemo(InteractiveDemo):
    def __init__(self,
                func: callable,
                x0: float = 0.0,
                N: int = 10,
                fig_num: int=None,
                figsize: tuple = (9,8)
                ):
        
        self.g = func

        
        self.x0 = x0
        self.N = N
        self.n = 0
        self.atol = 1e-8
        self.rtol = 1e-6
        self.x_points, self.y_points = self.fixpoint_iteration()
        self.xticklabels = ["$x_{%d}$"%(i) for i in range(self.N+1)]
        self.yticklabels = ["$g(x_{%d})$"%(i) for i in range(self.N)]
        
        super().__init__(fig_num=fig_num, figsize=figsize)

    def init_plot(self):
        ax = self.fig.add_subplot(1,1,1)
        a = min(self.x_points)
        b = max(self.x_points)
        a -= (b-a)*0.2
        b += (b-a)*0.2

        x = np.linspace(a, b, 501)
        self.x_line, = ax.plot(x, x, 'k', label="$x$")
        self.g_line, = ax.plot(x, self.g(x), 'C3', label="$g(x)$")
        self.cobweb, = ax.plot([self.x_points[0]], [self.y_points[0]], 'o--C0')
        ax.set_xlim([a, b])
        ax.set_ylim([a, b])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax_top = ax.secondary_xaxis('top')
        # Set custom ticks (example)
        ax.set_xticks([self.x_points[0]])
        ax.set_yticks([])
        ax.set_xticklabels(["$x_0$"])

        ax_right = ax.secondary_yaxis('right')
        ax.grid(True)
        ax.legend()
        ax.set_aspect(1)
        ax.set_title(f"$x_0 = {self.x0:.3f}$")
        #self.fig.tight_layout()
        self.ax = ax

        
    def create_widgets(self):
        self.n_slider = IntSlider(value=self.n,
                                  min=0,
                                  max=self.N,
                                  description='n',
                                  layout=Layout(width='70%'))

        self.n_slider.observe(lambda change: self.update_n(change['new']), names='value')
        self.layout = HBox([self.n_slider])

    def update_n(self, n):
        self.n = n
        self.cobweb.set_xdata(self.x_points[0:2*n+1])
        self.cobweb.set_ydata(self.y_points[0:2*n+1])

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_yticks(self.y_points[1:2*n:2])
        self.ax.set_yticklabels(self.yticklabels[0:n])

        self.ax.set_xticks(self.x_points[0:2*n+1:2])
        self.ax.set_xticklabels(self.xticklabels[0:n+1])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if n == 0:
            self.ax.set_title(f"$x_0 = {self.x0:.3f}$")
        else:
            self.ax.set_title("$x_{%d} = g(x_{%d}) \\approx g({%.3f}) \\approx {%.3f}$"%(n, n-1, self.x_points[2*n-1], self.y_points[2*n]))
        
    def fixpoint_iteration(self):
        x = self.x0
        x_points = [x]
        y_points = [x]
        gx = self.g(x)
        n = 0
        while np.abs(gx - x) > max(self.atol, self.rtol*np.abs(x)) and n < self.N:
            x_points.append(x)
            x_points.append(gx)
            y_points.append(gx)
            y_points.append(gx)
            x = gx
            gx = self.g(x)
            n += 1

        self.N = n
        return np.array(x_points), np.array(y_points)

