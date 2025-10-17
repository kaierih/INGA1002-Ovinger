import matplotlib.pyplot as plt
from ipywidgets import Output
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

