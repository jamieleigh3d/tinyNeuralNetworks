import wx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class ImageLossFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(2100, 1200))
        
        self.panel = wx.Panel(self)
        
        self.SetPosition((100, 10))  # Set X and Y coordinates
                
        self.figure1, self.plot1 = self.create_plot()
        self.figure2, self.plot2 = self.create_plot()
        self.figure3, self.plot3 = self.create_plot()
        self.canvas1 = FigureCanvas(self.panel, -1, self.figure1)
        self.canvas2 = FigureCanvas(self.panel, -1, self.figure2)
        self.canvas3 = FigureCanvas(self.panel, -1, self.figure3)
        
        self.rows = 3
        self.cols = 12
        # Using a FlexGridSizer for the image grid
        self.grid = wx.FlexGridSizer(self.rows, self.cols, 10, 10)
        for i in range(self.cols):
            self.grid.AddGrowableCol(i, 1)
        for i in range(self.rows):
            self.grid.AddGrowableRow(i, 1)
        
        # Create image placeholders
        self.image_boxes = [wx.StaticBitmap(self.panel) for _ in range(self.rows*self.cols)]  
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.EXPAND)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.canvas1, 1, wx.EXPAND)
        self.hbox.Add(self.canvas2, 1, wx.EXPAND)
        self.hbox.Add(self.canvas3, 1, wx.EXPAND)
        
        self.vbox.Add(self.hbox)
        
        self.panel.SetSizer(self.vbox)
        
        self.sc = None
        
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

    def create_plot(self):
        figure = Figure()
        plot = figure.add_subplot(111)
        return figure, plot
    
    def update_plot(self, total_losses, r_losses, kld_losses, d_losses):
        self.plot1.clear()
        self.plot1.plot(r_losses)
        self.plot1.plot(total_losses)
        
        #self.plot1.set_yscale('log')
        self.plot1.set_xlabel('Epoch')
        self.plot1.set_ylabel('Loss')
        self.plot1.set_title('Total/Recon Losses')
        self.canvas1.draw()
        
        self.plot2.clear()
        self.plot2.plot(kld_losses)
        
        self.plot2.set_xlabel('Epoch')
        self.plot2.set_ylabel('Loss')
        self.plot2.set_title('KLD Losses')
        self.canvas2.draw()        
        
        self.plot3.clear()
        self.plot3.plot(d_losses)
        
        self.plot3.set_xlabel('Epoch')
        self.plot3.set_ylabel('Loss')
        self.plot3.set_title('Perceptual Losses')
        self.canvas3.draw()
        

    def PIL_to_wxBitmap(self, pil_image):
        width, height = pil_image.size
        buffer = pil_image.convert("RGB").tobytes()
        wx_image = wx.Image(width, height, buffer)
        bitmap = wx_image.ConvertToBitmap()  # This converts it to a wx.Bitmap
        return bitmap

    def show_pca(self, latent_vectors):
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(latent_vectors)
    
        if self.sc is not None:
            self.sc.remove()
        self.sc = plt.scatter(pca_results[:, 0], pca_results[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA visualization of latent space')
        plt.draw()
        plt.show()
        
    def show_tsne(self, latent_vectors):
        seed=42
        tsne = TSNE(random_state=seed, n_components=2, perplexity=1, n_iter=300)
        tsne_results = tsne.fit_transform(latent_vectors)
    
        if self.sc is not None:
            self.sc.remove()
        self.sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE visualization of latent space')
        plt.draw()
        plt.show()

    def show_images(self, idx_images, total_losses, r_losses, kld_losses, d_losses, latent_vectors=None):
        
        self.update_plot(total_losses, r_losses, kld_losses, d_losses)
        
        for (idx, img) in idx_images:
            #width, height = (128,128)
            #img = img.resize((width, height))
            bitmap = self.PIL_to_wxBitmap(img)
            if idx >= 0 and idx < len(self.image_boxes):
                self.image_boxes[idx].SetBitmap(bitmap)
        
        if latent_vectors is not None:
            self.show_pca(latent_vectors)