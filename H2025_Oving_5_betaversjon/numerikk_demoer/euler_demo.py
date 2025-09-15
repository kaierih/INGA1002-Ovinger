from numpy import sin, cos, pi, exp, real, imag
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.signal as sig
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout, Output
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

def secant_solve(g: callable, x0: float, h: float = 1/1024, atol=1e-8, rtol=1e-6):
    x = x0
    g_x = g(x)
    while np.abs(g_x) > max(atol, rtol*x):
        dg_dx = (g(x+h)-g(x-h))/(2*h)
        x -= g_x/dg_dx
        g_x = g(x)
    return x

class EulerDemo1D(InteractiveDemo):
    """
    Class to create a interactive visualization of Eulers Method for first order 
    systems in 1 dimension.

    Parameters:
    func:   python-function describing the differential equation according to 
            y´ = f(x, y).
    y0:     initial value for y
    x0:     initial value for x
    xN:     last value of x for wich to compute y(x)
    """
    def __init__(self, 
                 func: callable, 
                 y0: float, 
                 x0: float = 0.0, 
                 xN: float = 1.0,
                 fig_num: int=None, 
                 figsize: tuple = (8,6)
                ):
        
        self.f = func
        self.N = 10
        self.h = (xN - x0)/self.N
        self.n = 0
        #x_N = x0 + self.N*h
        x_N = xN

        # Calculate approximate solution with Eulers Method
        self.x = np.linspace(x0, x0+self.N*self.h, self.N+1)
        self.y = self.get_euler_points(y0)

        # Calculate "exact" solution using Trapezoidal Integration and high resolution
        self.res = 500
        self.x_exact = np.linspace(x0, x_N, self.res+1)
        self.y_exact = self.get_trapezoidal_points(y0)

        super().__init__(fig_num = fig_num, figsize = figsize)

    def init_plot(self):
        ax = self.fig.add_subplot(1,1,1)
        ax.plot(self.x_exact, self.y_exact)
        self.euler_line, = ax.plot(self.x[0:self.n+1], self.y[0:self.n+1], "o-")
        # self.arrow = ax.quiver(self.x[self.n], 
        #                            self.y[self.n], 
        #                            self.x[self.n+1] - self.x[self.n], 
        #                            self.y[self.n+1] - self.x[self.n], 
        #                            angles='xy', color="tab:red", width=0.003)

        ax.set_xlim([self.x[0], self.x[-1]])
        ylim = ax.get_ylim()
        # Create a grid of x and y values
        x_arrow_points = np.linspace(self.x[0], self.x[-1], len(ax.get_xticks())*2-1)
        y_arrow_points = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())*2-1)
        
        X, Y = np.meshgrid(x_arrow_points,
                           y_arrow_points)
        
        # Compute the slope (dy/dx) at each point
        U = 1  # dx = 1 for unit horizontal vector
        V = np.zeros((len(y_arrow_points), len(x_arrow_points))) #self.f(X, Y)  # dy = f(x, y)
        for i, x in enumerate(x_arrow_points):
            for j, y in enumerate(y_arrow_points):
                V[j, i] = self.f(x,y)

        # Normalize the vectors for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / magnitude
        V_norm = V / magnitude
        
        # Plot the vector field using quiver
        ax.quiver(X, Y, U_norm, V_norm, angles='xy', color="tab:grey", width=0.003)
        ax.set_ylim([ylim[0], ylim[-1]])
        ax.grid(True)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y(x)$")
        self.ax = ax
        self.set_title()
        
    def create_widgets(self):
        self.n_slider = IntSlider(value=self.n,
                                  min=0,
                                  max=self.N,
                                  description='n',
                                  layout=Layout(width='50%'))
        
        self.h_select = widget.BoundedFloatText(value=self.h,
                                                min = (self.x_exact[-1] - self.x_exact[0])/100,
                                                max = (self.x_exact[-1] - self.x_exact[0])/2,
                                                step = (self.x_exact[-1] - self.x_exact[0])/100,
                                                description = 'h',
                                                layout=Layout(width='25%'))
        
        self.n_slider.observe(lambda change: self.update_n(change['new']), names='value')
        self.h_select.observe(lambda change: self.update_h(change['new']), names='value')

        self.layout = HBox([self.n_slider, self.h_select], layot=Layout(width='50%'))

    def update_n(self, n):
        self.n = n
        self.euler_line.set_xdata(self.x[0:n+1])
        self.euler_line.set_ydata(self.y[0:n+1])
        self.set_title()

    def update_h(self, h):
        self.h = h
        self.N = int((self.x_exact[-1] - self.x_exact[0])/h)
        self.n_slider.max = self.N
        if self.n > self.N:
            self.n = self.N
        self.x = np.linspace(self.x_exact[0], self.x_exact[0] + self.h*self.N, self.N +1)
        self.y = self.get_euler_points(self.y[0])
        self.euler_line.set_xdata(self.x[0:self.n+1])
        self.euler_line.set_ydata(self.y[0:self.n+1])
        self.set_title()

        
    def get_euler_points(self, y0: float):
        y = np.concatenate(([y0], np.zeros(self.N)))
        for n in range(self.N):
            y[n+1] = y[n] + self.h*self.f(self.x[n], y[n])
        return y
        
    def get_trapezoidal_points(self, y0: float):
        y = np.concatenate(([y0], np.zeros(self.res)))
        h_exact = (self.x_exact[-1] - self.x_exact[0])/self.res

        for n in range(self.res):
            k1 = self.f(self.x_exact[n], y[n])
            def g(k2):
                return self.f(self.x_exact[n+1], y[n] + h_exact/2*(k1 + k2)) - k2
            k2 = secant_solve(g, x0=y[n] + h_exact*k1)
            y[n+1] = y[n] + h_exact*(k1 + k2)/2
        return y

    def set_title(self):
        if self.n == 0:
            self.ax.set_title(r"$y_{%d} = %.2f$"%(self.n, self.y[0]))
        else:
            self.ax.set_title("$y_{%d} = y_{%d} + h \\cdot f(x_{%d}, y_{%d}) =  %.2f + %.2f \\cdot f(%.2f, %.2f) = %.2f$"%
                              (self.n, self.n-1, self.n-1, self.n-1, self.y[self.n-1], self.h, self.x[self.n-1], self.y[self.n-1], self.y[self.n]))

def f(t, v):
    ### BEGIN SOLUTION
    g = 9.81
    beta = 0.227
    m = 70
    if t < 0:
        v_diff = 0.0
    else:
        v_diff = g - beta*v**2/m
    ### END SOLUTION
    return v_diff

temp = EulerDemo1D(f, y0=0.0, x0=0.0, xN = 15.0, fig_num=1)
