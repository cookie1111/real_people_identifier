from torch.hub import load_state_dict_from_url

from models import *
from datasets import ImageCaptionDataset
from torchtext.models import XLMR_BASE_ENCODER
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from torchtext import transforms as text_transforms
from sklearn.metrics import f1_score, accuracy_score

# Hyperparameters
BATCH_SIZE = 256
NUM_WORKERS = 8
EPOCHS = 17
FOLDS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

im_transform = torch.nn.Sequential(
    transforms.Resize((224, 224), antialias=True),
)

xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
max_seq_len = 256
padding_idx = 1
bos_idx = 0
eos_idx = 2
text_transforms = text_transform = text_transforms.Sequential(
    text_transforms.SentencePieceTokenizer(xlmr_spm_model_path),
    text_transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    text_transforms.Truncate(max_seq_len - 2),
    text_transforms.AddToken(token=bos_idx, begin=True),
    text_transforms.AddToken(token=eos_idx, begin=False),
    text_transforms.ToTensor(padding_value=padding_idx),
    text_transforms.PadTransform(max_length=max_seq_len, pad_value=padding_idx),
)
ds = ImageCaptionDataset("params.json", transform_im=im_transform, transform_cap=text_transforms)

"""# Split dataset into training and validation sets
kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold = 0

for trainidx, testidx in kfold.split(ds):
    fold += 1
    print(f"Fold {fold}/{FOLDS}")

    # Create data loaders for the current fold
    train_loader = DataLoader(Subset(ds, trainidx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(ds, testidx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    resnet = init_resnet()
    roberta = init_roberta()
    model = ResNetRobertaEnsamble(num_classes=1, input_dim=256, resnet=resnet, roberta=roberta)
    model.to(DEVICE)

    # Define loss function and optimizer
    # Compute class weights
    class_weights = torch.tensor([len(ds) / (2 * (len(ds) - 2000)), len(ds) / (2 * 2000)]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    #gives epoch and validation accuracy
    best_acc_val = (0,0)
    best_f1 = (0,0)
    all_val_acc = []
    all_val_f1 = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_labels = []
        all_predictions = []
        for images, captions, labels in train_loader:
            images, captions, labels = images.to(DEVICE), captions.to(DEVICE), labels.to(DEVICE)
            captions = captions.squeeze(1)

            # Forward pass
            outputs = model(images, captions)
            loss = criterion(outputs.squeeze(), labels.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs)).squeeze().detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted)

        accuracy = accuracy_score(all_labels, all_predictions)
        train_f1 = f1_score(all_labels, all_predictions)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {accuracy:.4f}")

        # Validation loop
        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, captions, labels in val_loader:
                images, captions, labels = images.to(DEVICE), captions.to(DEVICE), labels.to(DEVICE)
                captions = captions.squeeze(1)
                outputs = model(images, captions)
                predicted = torch.round(torch.sigmoid(outputs)).squeeze().detach().cpu().numpy()

                # Store labels and predictions for F1 score computation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted)

            #print(f"Validation Accuracy: {100 * correct / total:.2f}%")
            val_f1 = f1_score(all_labels, all_predictions)
            accuracy = accuracy_score(all_labels, all_predictions)
            print(f"Validation F1 for Fold {fold}: {val_f1:.4f}, Validation Accuracy: {accuracy:.4f}")
            all_val_f1.append(val_f1)
            all_val_acc.append(accuracy)
            if val_f1 > best_f1[1]:
                best_f1 = (epoch, val_f1)
            if accuracy > best_acc_val[1]:
                best_acc_val = (epoch, accuracy)
    avg_val_accuracy = sum(all_val_acc)/epoch
    avg_val_f1 = sum(all_val_f1)/epoch
    print(f"\nAverage Validation Accuracy across all epochs: {avg_val_accuracy:.4f}")
    print(f"Average Validation F1 Score across all epochs: {avg_val_f1:.4f}")
    print(f"Best F1 Score: {best_f1[1]:.4f} at Epoch {best_f1[0]}"
          f"\nBest Validation Accuracy: {best_acc_val[1]:.4f} at Epoch {best_acc_val[0]}")
"""

train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
resnet = init_resnet()
roberta = init_roberta()
model = ResNetRobertaEnsamble(num_classes=1, input_dim=256, resnet=resnet, roberta=roberta)
model.to(DEVICE)


class_weights = torch.tensor([len(ds) / (2 * (len(ds) - 2000)), len(ds) / (2 * 2000)]).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    for images, captions, labels in train_loader:
        images, captions, labels = images.to(DEVICE), captions.to(DEVICE), labels.to(DEVICE)
        captions = captions.squeeze(1)

        # Forward pass
        outputs = model(images, captions)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.round(torch.sigmoid(outputs)).squeeze().detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted)

    accuracy = accuracy_score(all_labels, all_predictions)
    train_f1 = f1_score(all_labels, all_predictions)

    print(
        f"Epoch [{epoch}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'whole_set_train.pth')
print("Model saved successfully!")