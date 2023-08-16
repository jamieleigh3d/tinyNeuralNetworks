import wx
import random
import threading
import time
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

class LossGraphApp(wx.App):
    def OnInit(self):
        self.frame = LossGraphFrame()
        self.frame.Show()
        return True

class LossGraphFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title='Loss Graph', size=(600, 400))
        self.losses = []
        self.panel = wx.Panel(self)
        self.figure = self.create_plot()
        self.canvas = FigureCanvas(self.panel, -1, self.figure)  # Use the panel as the parent
        
        self.start_button = wx.Button(self.panel, label='Start Training')
        self.start_button.Bind(wx.EVT_BUTTON, self.start_training)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)
        self.sizer.Add(self.start_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        self.panel.SetSizer(self.sizer)
    
    def create_plot(self):
        figure = Figure()
        self.plot = figure.add_subplot(111)
        self.plot.set_xlabel('Epoch')
        self.plot.set_ylabel('Loss')
        self.plot.set_title('Loss Graph')
        return figure
    
    def update_plot(self):
        self.plot.clear()
        self.plot.plot(self.losses)
        self.canvas.draw()
    
    def start_training(self, event):
        self.start_button.Disable()
        self.training_thread = threading.Thread(target=self.simulate_training)
        self.training_thread.start()
    
    def simulate_training(self):
        for epoch in range(1, 11):
            loss = random.uniform(0.1, 1.0)  # Simulated loss value
            self.losses.append(loss)
            
            wx.CallAfter(self.update_plot)  # Update the plot on the GUI thread
            
            time.sleep(1)  # Simulate training time
            
        wx.CallAfter(self.start_button.Enable)

if __name__ == '__main__':
    app = LossGraphApp()
    app.MainLoop()
