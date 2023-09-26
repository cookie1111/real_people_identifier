import pandas as pd
from torch.hub import load_state_dict_from_url
import os
from models import *
from datasets import ImageCaptionDataset
from torchvision.transforms import transforms
import torch
from torchtext import transforms as text_transforms
from sklearn.metrics import f1_score, accuracy_score
import json
from PIL import Image
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 256
NUM_WORKERS = 8
EPOCHS = 20
FOLDS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
im_tensor = transforms.ToTensor()
im_transform = torch.nn.Sequential(
    transforms.Resize((224, 224), antialias=True),
)

xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
max_seq_len = 256
padding_idx = 1
bos_idx = 0
eos_idx = 2
text_transform = text_transforms.Sequential(
    text_transforms.SentencePieceTokenizer(xlmr_spm_model_path),
    text_transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    text_transforms.Truncate(max_seq_len - 2),
    text_transforms.AddToken(token=bos_idx, begin=True),
    text_transforms.AddToken(token=eos_idx, begin=False),
    text_transforms.ToTensor(padding_value=padding_idx),
    text_transforms.PadTransform(max_length=max_seq_len, pad_value=padding_idx),
)
main_folder = json.load(open("params.json", 'r'))["params"]["data_path"]
img_folder = os.path.join(main_folder, 'img')
caption_folder = os.path.join(main_folder, 'captions')
folders = ["test", "train", "val"]
cities = ['chicago', 'losangeles', 'miami', 'london', 'melbourne', 'newyork', 'sanfrancisco', 'singapore', 'sydney', 'toronto']

dfs = {x: pd.DataFrame(
    columns=["ID", "city", "label"]) for x in folders}

resnet = init_resnet()
roberta = init_roberta()
model = ResNetRobertaEnsamble(num_classes=1, input_dim=256, resnet=resnet, roberta=roberta)
model.load_state_dict(torch.load('whole_set_train.pth'))
model.eval()
model.to(DEVICE)

# Validation loop
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for folder in folders:
        print(f"Processing cities in the {folder} set")
        for city in cities:
            img_path = os.path.join(img_folder, folder, city)
            caption_path = os.path.join(caption_folder, folder, city)

            total_images = len([f for f in os.listdir(img_path) if f.endswith('.jpg')])
            pbar = tqdm(total=total_images, desc=f"Processing images in {city}", ncols=100)

            for i, filename in enumerate(sorted(os.listdir(img_path))):
                if filename.endswith('.jpg'):
                    caption = text_transform(
                        open(os.path.join(caption_path, filename[:-4] + '.txt'), 'r').read().split("\n"))
                    image = im_transform(im_tensor((Image.open(os.path.join(img_path, filename)).convert("RGB"))))
                    image = torch.unsqueeze(image, 0)
                    image, caption = image.to(DEVICE), caption.to(DEVICE)
                    output = model(image, caption)
                    output = torch.round(torch.sigmoid(output)).squeeze().detach().cpu().numpy()
                    dfs[folder] = dfs[folder].append({"ID": filename[:-4], "city": city, "label": output.item()}, ignore_index=True)
                    pbar.update(1)

            pbar.close()



for folder in folders:
    dfs[folder].to_csv(f"ouput_{folder}.csv")




