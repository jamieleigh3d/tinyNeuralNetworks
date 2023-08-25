import wx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class ImageLossFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(2100, 1200))
        
        self.SetDoubleBuffered(True)
        
        self.SetPosition((100, 10))  # Set X and Y coordinates
        
        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 4))
        self.canvas = FigureCanvas(self, -1, self.fig)
        
        for ax in self.axes:
            ax.axis('off')
            
        self.rows = 4
        self.cols = 12
        # Using a FlexGridSizer for the image grid
        self.grid = wx.FlexGridSizer(self.rows, self.cols, 10, 10)
        for i in range(self.cols):
            self.grid.AddGrowableCol(i, 1)
        for i in range(self.rows):
            self.grid.AddGrowableRow(i, 1)
        
        # Create image placeholders
        self.image_boxes = [wx.StaticBitmap(self) for _ in range((self.rows-1)*self.cols)]  
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.EXPAND)
            
        self.labels = [wx.StaticText(self, label="") for _ in range(self.cols)]
        self.label_width = 150
        for label in self.labels:
            label.Wrap(self.label_width)
            self.grid.Add(label, flag=wx.EXPAND)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.vbox.Add(self.canvas)
        
        self.SetSizer(self.vbox)
        
        self.sc = None
        
        self.Layout()
        
        self.done = False
        self.thread = None
        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        # Bind the resize event
        self.Bind(wx.EVT_SIZE, self.OnResize)
        
    def OnClose(self, event):
        # Do cleanup here, stop threads, release resources
        self.Destroy()
        wx.GetApp().ExitMainLoop()
        self.done = True
        if self.thread is not None:
            self.thread.join()
            self.thread = None
 
    def OnResize(self, event):
        # Get the new width of the window
        new_width = self.GetSize().width

        # For demonstration, let's say you want to wrap the labels at half the window width
        # Adjust this calculation as necessary
        padding = 5
        self.label_width = (new_width-padding) / self.cols
        
        for label in self.labels:
            label.Wrap(self.label_width)

        # Make sure the layout and display are updated
        self.Layout()

        # Continue processing the event
        event.Skip()
        
    def create_plot(self):
        figure = Figure()
        plot = figure.add_subplot(111)
        return figure, plot
    
    def update_plot(self, total_losses, avg_losses, r_losses, learning_rates, p_losses):
        ax0 = self.axes[0]
        ax0.clear()
        if r_losses is not None:
            ax0.plot(r_losses)
        if total_losses is not None:
            ax0.plot(total_losses)
        if avg_losses is not None:
            ax0.plot(avg_losses)
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Loss')
        ax0.set_title('Total/Recon Losses')
        
        ax1 = self.axes[1]
        ax1.clear()
        ax1.clear()
        if learning_rates is not None:
            ax1.plot(learning_rates)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('LR')
        ax1.set_title('Learning Rate')
        
        ax2 = self.axes[2]
        ax2.clear()
        if p_losses is not None:
            ax2.plot(p_losses)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Perceptual Losses')
        

    def PIL_to_wxBitmap(self, pil_image):
        width, height = pil_image.size
        buffer = pil_image.convert("RGB").tobytes()
        wx_image = wx.Image(width, height, buffer)
        bitmap = wx_image.ConvertToBitmap()  # This converts it to a wx.Bitmap
        return bitmap

    def show_pca(self, latent_vectors):
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(latent_vectors)
    
        ax = self.axes[3]
        ax.clear()
    
        ax.scatter(pca_results[:, 0], pca_results[:, 1], s=2)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA visualization of latent space')
        
        
    def show_tsne(self, latent_vectors):
        seed=42
        tsne = TSNE(random_state=seed, n_components=2, perplexity=1, n_iter=300)
        tsne_results = tsne.fit_transform(latent_vectors)
    
        ax = self.axes[3]
        ax.clear()
    
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=2)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set+title('t-SNE visualization of latent space')
        

    def show_images(self, idx_images, label_pairs=None, total_losses=None, avg_losses=None, r_losses=None, learning_rates=None, p_losses=None, latent_vectors=None):
        
        self.update_plot(total_losses, avg_losses, r_losses, learning_rates, p_losses)
        
        self.Freeze()
        
        if label_pairs is not None:
            for (idx, text) in label_pairs:
                self.labels[idx].SetLabel(text)
                self.labels[idx].Wrap(self.label_width)
        
            # Make sure the layout and display are updated
            #self.Layout()
        
        self.Thaw()
        for (idx, img) in idx_images:
            width, height = (128,128)
            if img.width < width or img.height < height:
                img = img.resize((width, height))
            bitmap = self.PIL_to_wxBitmap(img)
            if idx >= 0 and idx < len(self.image_boxes):
                self.image_boxes[idx].SetBitmap(bitmap)
        
        if latent_vectors is not None:
            self.show_pca(latent_vectors)
        
        self.canvas.draw()
