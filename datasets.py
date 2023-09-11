from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import json
from PIL import Image
from torchvision.transforms import transforms


class ImageCaptionDataset(Dataset):

    def __init__(self, params, local_dataset="output.csv", ds_type="train", transform_im=None, transform_cap=None):
        self.ds_type = ds_type
        self.data_path=json.load(open(params, 'r'))["params"]["data_path"]
        self.dataset = pd.read_csv(local_dataset)
        self.to_tensor = transforms.ToTensor()
        self.transform_im = transform_im
        self.transform_cap = transform_cap
        self.captions = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        doc_id = str(self.dataset.ID.iloc[idx])
        location = self.dataset.City.iloc[idx]
        caption_path = self.data_path + "/captions/"+self.ds_type+"/"+location+"/"+doc_id+".txt"
        caption = open(caption_path, "r").read().split("\n")
        image_path = self.data_path + "/img/"+self.ds_type+"/"+location+"/"+doc_id+".jpg"

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        clas = self.dataset.Person.iloc[idx]

        if self.transform_im:
            image = self.transform_im(image)

        if self.transform_cap:
            caption = self.transform_cap(caption)

        return image, caption, clas


#testing
im_transform = torch.nn.Sequential(
    transforms.Resize((224,224),antialias=True),
)
ds = ImageCaptionDataset("params.json")
dl = DataLoader(ds, batch_size=8, shuffle=True)
print(len(ds))
print(ds[0])

for i in dl:
    print(i)