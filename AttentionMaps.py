import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class AttentionMapsFrame(wx.Frame):
    def __init__(self, parent, title, num_layers, num_heads):
        super(AttentionMapsFrame, self).__init__(parent, title=title, size=(1800, 1200))
        
        self.SetPosition((100, 10))  # Set X and Y coordinates
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fig, self.axes = plt.subplots(num_heads, num_layers, figsize=(15, 10))
        self.canvas = FigureCanvas(self, -1, self.fig)
        
        for ax0 in self.axes:
            for ax in ax0:
                ax.axis('off')
            
        plt.tight_layout()
        self.Layout()
        
        self.done = False
        self.thread = None
        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
        
    def OnClose(self, event):
        # Do cleanup here, stop threads, release resources
        
        self.Destroy()
        wx.GetApp().ExitMainLoop()
        self.done = True
        if self.thread is not None:
            self.thread.join()
        
    def plot_attention_map_for_head(self, attns, image, layer, head, ax):
        # Ensure the image is between [0, 1]
        image = image.permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.squeeze(-1)

        ax.clear()
        # Convert attention map to numpy
        attn_map = attns[0, layer, head].cpu().numpy()
        
        # Ignore the CLS token
        attn_map = attn_map[1:, 1:]
        
        # Upsample the attention map to be the same size as the image
        H, W = image.shape
        zoom_factor = (H / attn_map.shape[0], W / attn_map.shape[1])
        attn_map_resized = zoom(attn_map, zoom_factor)
        
        # Plot the image and the attention map
        ax.imshow(image, cmap='gray')
        ax.imshow(attn_map_resized, cmap='hot', alpha=0.5)
        ax.set_title(f"Layer {layer + 1}, Head {head + 1}")
        ax.axis('off')
        

    def plot_attention_maps_for_all(self, attns, image_sample):
        # Visualizes attention maps for all layers and heads in subplots.
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                ax = self.axes[head, layer]
                self.plot_attention_map_for_head(attns, image_sample, layer, head, ax)
        self.canvas.draw()

    # Function called in UI thread via wx.CallAfter()
    def show_attns(self, attns, image_sample):
        
        self.plot_attention_maps_for_all(attns, image_sample)
