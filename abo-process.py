import json
import csv
import os
import wx
import time
import threading

domain_tag = 'domain_name'
main_image_id_tag = 'main_image_id'
item_id_tag = 'item_id'
item_name_tag = 'item_name'
image_id_tag = 'image_id'
height_tag = 'height'
width_tag = 'width'
path_tag = 'path'

listings_filepath = "data/abo/listings/metadata/listings_0.json"
imagedata_filepath = "data/abo/images/metadata/images.csv"

data = []

def filter(json_obj):
    if domain_tag in json_obj:
        return json_obj[domain_tag] == 'amazon.com'
    return False
   
with open(listings_filepath, 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        if filter(json_obj):
            data.append(json_obj)

image_list = {}
with open (imagedata_filepath, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        if image_id_tag in row:
            image_id = row[image_id_tag]
            image_list[image_id] = row

print(len(data))
obj = data[0]

item_id = obj.get(item_id_tag, "<UNK>")
item_name = obj.get(item_name_tag, "[{}]")[0].get('value', "<UNK>")
main_image_id = obj.get(main_image_id_tag, "<UNK>")
print(item_id)
print(item_name)
print(main_image_id)

main_image = image_list[main_image_id]
print(main_image)


def get_filepath_for_object(obj, image_list):
    main_image_id = obj.get(main_image_id_tag, "<UNK>")
    main_image = image_list[main_image_id]
    image_folder = "data/abo/images/small"
    image_filepath = os.path.join(image_folder, main_image[path_tag])
    return image_filepath
    

class ImageLoaderFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1200, 800))
        
        self.panel = wx.Panel(self)
        
        # Using a FlexGridSizer for the image grid
        self.grid = wx.FlexGridSizer(3, 3, 10, 10)
        self.grid.AddGrowableCol(0, 1)
        self.grid.AddGrowableCol(1, 1)
        self.grid.AddGrowableCol(2, 1)
        self.grid.AddGrowableRow(0, 1)
        self.grid.AddGrowableRow(1, 1)
        #self.grid.AddGrowableRow(2, 1)
        
        self.image_boxes = [wx.StaticBitmap(self.panel) for _ in range(9)]  # Create 9 image placeholders
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.EXPAND)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.panel.SetSizer(self.vbox)
        
        self.start_loading()
        
    def load_and_display_image(self):
        for i, obj in enumerate(data[:6]):
            img_path = get_filepath_for_object(obj, image_list)
            
            time.sleep(1)  # Simulate image processing
            img = wx.Image(img_path, wx.BITMAP_TYPE_ANY)#.Scale(140, 140)  # Load and scale the image
            self.image_boxes[i].SetBitmap(wx.Bitmap(img))
            self.image_boxes[i].Refresh()
            
    def start_loading(self):
        t = threading.Thread(target=self.load_and_display_image, args=())
        t.start()


if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageLoaderFrame(None, 'Image Processing GUI')
    frame.Show()
    app.MainLoop()
    
print('Done!')