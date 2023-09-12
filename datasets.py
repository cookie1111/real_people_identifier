from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import json
from PIL import Image
from torchtext.models import XLMR_BASE_ENCODER
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchtext.transforms as text_transforms


class ImageCaptionDataset(Dataset):

    def __init__(self, params, local_dataset="output.csv", ds_type="train", transform_im=None, transform_cap=None,
                 max_tokens=256):
        self.ds_type = ds_type
        self.data_path=json.load(open(params, 'r'))["params"]["data_path"]
        self.dataset = pd.read_csv(local_dataset)
        self.to_tensor = transforms.ToTensor()
        self.transform_im = transform_im
        self.transform_cap = transform_cap
        self.max_tokens = max_tokens
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
        #-print(caption)
        if self.transform_im:
            image = self.transform_im(image)

        if self.transform_cap:
            caption = self.transform_cap(caption)

        return image, caption, clas

    def info(self):
        self.dataset.Person.value_counts().plot.bar()
        plt.show()


#testing
im_transform = torch.nn.Sequential(
    transforms.Resize((224,224),antialias=True),
)
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
max_seq_len = 256
padding_idx = 1
bos_idx = 0
eos_idx = 2
#text_transforms = XLMR_BASE_ENCODER.transform()
text_transform = text_transforms.Sequential(
    text_transforms.SentencePieceTokenizer(xlmr_spm_model_path),
    text_transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    text_transforms.Truncate(max_seq_len - 2),
    text_transforms.AddToken(token=bos_idx, begin=True),
    text_transforms.AddToken(token=eos_idx, begin=False),
    text_transforms.ToTensor(padding_value=padding_idx),
    text_transforms.PadTransform(max_length=max_seq_len,pad_value=padding_idx),
)
"""
#print(text_transform("today is a beautiful day"))
ds = ImageCaptionDataset("params.json", transform_im=im_transform, transform_cap=text_transform)
ds.info()
dl = DataLoader(ds, batch_size=32, shuffle=True)
print(len(ds))
print(len(ds[4][1][0]))
for i in dl:
    print(i[1].shape)
    #for j in range(len(i)):
    #    print(len(i[j][1]))
"""

