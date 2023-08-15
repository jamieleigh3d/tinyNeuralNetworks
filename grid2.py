import wx
import threading
import time

print('1')

# Create a list of image paths (replace these paths with your images)
image_paths = ["data/abo/images/small/07/075e5d67.jpg", 
                "data/abo/images/small/c6/c6889ed4.jpg", 
                "data/abo/images/small/2b/2b90e918.jpg", 
                "data/abo/images/small/2c/2c0416de.jpg", 
                "data/abo/images/small/ea/ea0c6da6.jpg"]


class ImageLoaderFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(800, 600))
        
        self.panel = wx.Panel(self)
        self.grid = wx.GridSizer(3, 3, 10, 10)  # 3x3 grid
        
        self.image_boxes = [wx.StaticBitmap(self.panel) for _ in range(9)]  # Create 9 image placeholders
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.ALIGN_CENTER)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.panel.SetSizer(self.vbox)
        
        self.start_loading()
        
    def load_and_display_image(self, i, img_path, box):
        time.sleep(i)  # Simulate image processing
        img = wx.Image(img_path, wx.BITMAP_TYPE_ANY)#.Scale(140, 140)  # Load and scale the image
        box.SetBitmap(wx.Bitmap(img))
        box.Refresh()
        
    def start_loading(self):
        for i, img_path in enumerate(image_paths):
            t = threading.Thread(target=self.load_and_display_image, args=(i, img_path, self.image_boxes[i]))
            t.start()

if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageLoaderFrame(None, 'Image Processing GUI')
    frame.Show()
    app.MainLoop()