from numpy import sin, cos, pi, exp, real, imag
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.signal as sig
from ipywidgets import IntSlider, HBox, Layout, Output, Button
import ipywidgets as widget
from IPython.display import display
from . import InteractiveDemo


def newton_method(f, df_dx, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding roots of a real-valued function.

    Parameters:
    -----------
    f : function
        The function for which we want to find a root.
    df_dx : function
        The derivative of f.
    x0 : float
        Initial guess for the root.
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-7).
    max_iter : int, optional
        Maximum number of iterations (default: 100).

    Returns:
    --------
    x : float
        Estimated root.
    iterations : int
        Number of iterations performed.
    """
    x = [x0]
    for i in range(max_iter):
        f_val = f(x[-1])
        df_val = df_dx(x[-1])

        if df_val == 0:
            return x, "zero_division"

        x_new = x[i] - f_val / df_val

        if abs(x_new - x[i]) < tol:
            return x, "converged"

        x.append(x_new)

    return np.array(x), "timeout"


class NewtonsMethodDemo(InteractiveDemo):
    def __init__(self,
                f: callable,
                df_dx: callable,
                x0: float = 0.0, 
                tol: float = 1e-6,
                xlim: list = None,
                fig_num: int = None,
                figsize: tuple = (9,6)
                ):
        
        self.f = f
        self.df_dx = df_dx
        self.n = 0
        if self.f(x0) == 0:
            print("Initial guess is already a root.")
            return
        else:
            self.x_points, status = newton_method(f, df_dx, x0, tol)

        self.tangent_func = lambda x: self.f(x0) + self.df_dx(x0)*(x-x0)
        
        if xlim is None:
            if status == "converged" or status == "zero_division":
                x_span = [x0, self.x_points[-1]]
                x_span.sort()
                x_dist = x_span[-1] - x_span[0]
                x_lim = [x_span[0] - 0.2*x_dist, x_span[-1] + 0.2*x_dist]
                if max(self.x_points) > x_lim[-1]:
                    x_lim[-1] = min(max(self.x_points)+x_dist*0.01, x_lim[-1] + 2*x_dist)
                if min(self.x_points) < x_lim[0]:
                    x_lim[0] = max(min(self.x_points)-x_dist*0.01, x_lim[0] - 2*x_dist)
            
            elif status == "timeout":
                x_list = self.x_points.copy()
                x_list.sort()
                x_span = [x_list[10], x_list[89]]
                x_dist = x_span[-1] - x_span[0]
                x_lim = [min(x0, x_span[0]) - 0.1*x_dist, max(x0, x_span[1]) + 0.1*x_dist]
                
            if status == "zero_division":
                self.x_points = np.concatenate((self.x_points, [float("inf")])) # Add a "false" point out of bounds
            x_lim[0] = min(x_lim[0], x0 - 1)
            x_lim[1] = max(x_lim[1], x0 + 1)
            self.xlim = np.array(x_lim)
        else:
            self.xlim = np.array(xlim)
        
        super().__init__(fig_num=fig_num, figsize=figsize)

    def init_plot(self):

        x = np.linspace(self.xlim[0], self.xlim[1], 501)

        
        ax = self.fig.add_subplot(1,1,1)
        ax.axhline(0, color='k', lw=0.8)
        self.f_line, = ax.plot(x, self.f(x), 'C0', label="$f(x)x$")
        self.tangent_line, = ax.plot(self.xlim, [0,0], 'C2:')
        self.newton, = ax.plot([self.x_points[0]], [self.f(self.x_points[0])], '.C3')
        self.step_points, = ax.plot(self.x_points[0:2], [self.f(self.x_points[0]), 0], 'o', markersize=8, markerfacecolor='none', markeredgecolor="tab:green", markeredgewidth=2.5)
        self.tangent_line.set_ydata(self.tangent_func(self.xlim))
        

        ax.set_xlim(self.xlim)
        self.ylim = ax.get_ylim()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax_top = ax.secondary_xaxis('top')
        # Set custom ticks (example)

        ax_right = ax.secondary_yaxis('right')

        ax.grid(True)

        #ax.set_title(f"$x_0 = {self.x0:.3f}$")
        self.ax = ax
        self.set_title()
        self.set_ticks()
        self.fig.tight_layout(rect=(0,0.02,0.95,1))

            
    def create_widgets(self):
        self.prev_btn = Button(description = "Forrige Steg",
                          disabled=True,
                          button_style = '')
        self.next_btn = Button(description = "Neste Steg",
                          disabled=False if len(self.x_points)>2 else True,
                          button_style = '')

        self.refocus_btn = Button(description = "Zoom inn/ut",
                                  disabled=False,
                                  button_style = '')
        
        self.restart_btn = Button(description = "Start på nytt",
                                  disabled=False,
                                  button_style = '')
        
        
        self.prev_btn.on_click(self.prev_step)
        self.next_btn.on_click(self.next_step)
        self.refocus_btn.on_click(self.refocus)
        self.restart_btn.on_click(self.restart)

        self.layout = HBox([self.restart_btn, self.prev_btn, self.next_btn, self.refocus_btn])

    def next_step(self, btn):
        self.n +=1
        self.update_plot()
        if self.n > 0:
            self.prev_btn.disabled = False
        if self.n >= len(self.x_points)-2:
            self.next_btn.disabled = True
            
    def prev_step(self, btn):
        self.n -=1
        self.update_plot()
        if self.n <= 0:
            self.prev_btn.disabled = True
        if self.n < len(self.x_points)-2:
            self.next_btn.disabled = False

    def update_plot(self):
        self.newton.set_data(self.x_points[:self.n+1], self.f(np.array(self.x_points[:self.n+1])))
        self.tangent_line.set_ydata(self.f(self.x_points[self.n]) + self.df_dx(self.x_points[self.n])*(self.tangent_line.get_xdata()-self.x_points[self.n]))
        self.step_points.set_data(self.x_points[self.n:self.n+2], [self.f(self.x_points[self.n]), 0])
        self.set_title()
        self.set_ticks()
        self.fig.canvas.draw_idle()

    def set_ticks(self):
        self.ax.set_xticks(self.x_points[self.n:self.n+2])
        self.ax.set_xticklabels(["$x_{%d}$"%(n) for n in range(self.n, self.n+2)])
        if min(self.x_points[self.n:self.n+2]) < self.xlim[0]:
            self.ax.set_xlim(xmin=self.xlim[0])
        if max(self.x_points[self.n:self.n+2]) > self.xlim[1]:
            self.ax.set_xlim(xmax=self.xlim[1])

        self.ax.set_yticks([self.f(self.x_points[self.n])])
        self.ax.set_yticklabels(["$f(x_{%d})$"%(self.n)])

        if self.f(self.x_points[self.n]) < self.ylim[0]:
            self.ax.set_ylim(ymin=self.ylim[0])
        if self.f(self.x_points[self.n]) > self.ylim[1]:
            self.ax.set_ylim(ymax=self.ylim[1])




    def set_title(self):
        title = """$x_{%d} = x_{%d} - \\frac{f(x_{%d})}{f'(x_{%d})} = %.3f - \\frac{f(%.3f)}{f'(%.3f)} = %.3f$ """ % (self.n+1, self.n, self.n, self.n, self.x_points[self.n], self.x_points[self.n], self.x_points[self.n], self.x_points[self.n+1]) 
        self.ax.set_title(title)


    def constrain_lim(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.fig.canvas.draw_idle()

    def refocus(self, btn):
        if self.n==0:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
        else:
            yvals = [self.f(x) for x in self.x_points[self.n-1:self.n+2]]
            ymin = np.min([0, np.min(yvals)])
            ymax = np.max([0, np.max(yvals)])
            y_dist = ymax - ymin
            ylim = [ymin - 0.15*y_dist, ymax + 0.15*y_dist]
            ylim.sort()
            self.ax.set_ylim(ylim)
            x_span = list(self.x_points[self.n-1:self.n+2])
            x_span.sort() 
            x_span.pop(1)
            x_dist = x_span[-1] - x_span[0]
            x_lim = [x_span[0] - 0.2*x_dist, x_span[-1] + 0.2*x_dist]
            if x_lim[0] < self.xlim[0]:
                x_lim[0] = self.xlim[0]
            if x_lim[1] > self.xlim[1]:
                x_lim[1] = self.xlim[1]
            self.ax.set_xlim(x_lim)
        self.fig.canvas.draw_idle()

    def restart(self, btn):
        self.n = 0
        self.prev_btn.disabled = True
        self.next_btn.disabled = False
        self.update_plot()
        self.refocus(None)
