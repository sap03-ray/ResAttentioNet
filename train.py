import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def per_subject_normalization(X, num_subjects, trials_per_subject):
    X_norm = np.zeros_like(X)
    for subj in range(num_subjects):
        start = subj * trials_per_subject
        end = (subj + 1) * trials_per_subject
        X_subj = X[start:end]
        scaler = StandardScaler()
        X_flat = X_subj.reshape(-1, X_subj.shape[-1])
        X_scaled = scaler.fit_transform(X_flat)
        X_norm[start:end] = X_scaled.reshape(X_subj.shape)
    return X_norm

def subject_independent_split(X, y, num_subjects, trials_per_subject, num_classes, train_split=0.7, val_split=0.15, max_attempts=100):
    classes = np.unique(y)
    assert len(classes) == num_classes
    subjects = list(range(num_subjects))
    for attempt in range(max_attempts):
        np.random.shuffle(subjects)
        train_idx_end = int(train_split * num_subjects)
        val_idx_end = train_idx_end + int(val_split * num_subjects)
        train_subjects = subjects[:train_idx_end]
        val_subjects = subjects[train_idx_end:val_idx_end]
        test_subjects = subjects[val_idx_end:]
        train_idx = np.concatenate([np.arange(s * trials_per_subject, (s + 1) * trials_per_subject) for s in train_subjects])
        val_idx = np.concatenate([np.arange(s * trials_per_subject, (s + 1) * trials_per_subject) for s in val_subjects])
        test_idx = np.concatenate([np.arange(s * trials_per_subject, (s + 1) * trials_per_subject) for s in test_subjects])
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        if (len(np.unique(y_train)) == num_classes and 
            len(np.unique(y_val)) == num_classes and 
            len(np.unique(y_test)) == num_classes):
            return X[train_idx], y_train, X[val_idx], y_val, X[test_idx], y_test
    raise ValueError("Could not find a valid split after {} attempts. Try increasing subjects or reducing class constraints.".format(max_attempts))

def print_class_distributions(y_train, y_val, y_test):
    print("Train classes:", np.unique(y_train, return_counts=True))
    print("Validation classes:", np.unique(y_val, return_counts=True))
    print("Test classes:", np.unique(y_test, return_counts=True))

def get_class_weights(y_train, num_classes):
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    return class_weights

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_training_summary(train_acc, val_acc, train_loss, val_loss, TP_list, FP_list, TN_list, FN_list, precision_list, recall_list, epoch):
    epochs = range(1, len(train_acc) + 1)
    smooth_train_acc = gaussian_filter1d(train_acc, sigma=2)
    smooth_val_acc = gaussian_filter1d(val_acc, sigma=2)
    smooth_train_loss = gaussian_filter1d(train_loss, sigma=2)
    smooth_val_loss = gaussian_filter1d(val_loss, sigma=2)
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_acc, color='blue', alpha=0.4)
    plt.plot(epochs, smooth_train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc, color='orange', alpha=0.4)
    plt.plot(epochs, smooth_val_acc, label='Val Accuracy', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"accuracy.png", dpi=300)
    plt.close()
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_loss, color='green', alpha=0.4)
    plt.plot(epochs, smooth_train_loss, label='Train Loss', color='green')
    plt.plot(epochs, val_loss, color='red', alpha=0.4)
    plt.plot(epochs, smooth_val_loss, label='Val Loss', color='red')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss.png", dpi=300)
    plt.close()
    plt.figure(figsize=(7,5))
    plt.plot(epochs, precision_list, label='Precision')
    plt.plot(epochs, recall_list, label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"precision_recall.png", dpi=300)
    plt.close()

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def make_layer(block, in_planes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv1d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * block.expansion),
        )
    layers = []
    layers.append(block(in_planes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(block(planes * block.expansion, planes))
    return nn.Sequential(*layers)

class MTAP(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.mha(query, x, x)
        return attn_output.squeeze(1)

class RTFE(nn.Module):
    def __init__(self, in_channels, base_width=32):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv1d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(ResNetBlock, base_width, base_width, blocks=2)
        self.layer2 = make_layer(ResNetBlock, base_width, base_width*2, blocks=2, stride=2)
        self.layer3 = make_layer(ResNetBlock, base_width*2, base_width*4, blocks=2, stride=2)
        self.layer4 = make_layer(ResNetBlock, base_width*4, base_width*8, blocks=2, stride=2)
        self.out_dim = base_width*8
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(1,2)
        return x

class EmotionResNetAttention(nn.Module):
    def __init__(self, num_features=310, seq_len=60, num_classes=NUM_CLASSES, base_width=16):
        super().__init__()
        self.encoder = RTFE(in_channels=num_features, base_width=base_width)
        self.dropout = nn.Dropout(0.5)
        self.attn_pool = MTAP(self.encoder.out_dim)
        self.edc = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.attn_pool(x)
        return self.edc(x)

def train_eval_loop(model, train_loader, val_loader, test_loader, epochs=100, lr=1e-3, class_weights=None, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    best_val_acc = 0
    patience, patience_counter = 10, 0
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    precision_list, recall_list = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        mcm = multilabel_confusion_matrix(all_labels, all_preds, labels=np.unique(all_labels))
        TP = sum(cm[1, 1] for cm in mcm)
        FP = sum(cm[0, 1] for cm in mcm)
        TN = sum(cm[0, 0] for cm in mcm)
        FN = sum(cm[1, 0] for cm in mcm)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        prec, rec, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        precision_list.append(prec)
        recall_list.append(rec)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    return train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, TP_list, FP_list, TN_list, FN_list, precision_list, recall_list

set_seed()
with open('/path/to/data.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['X']
y = data['y']
X_normed = per_subject_normalization(X, num_subjects, trials_per_subject)
X_train, y_train, X_val, y_val, X_test, y_test = subject_independent_split(X_normed, y, num_subjects, trials_per_subject, num_classes=num_classes, train_split=0.7, val_split=0.15)
print_class_distributions(y_train, y_val, y_test)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_weights = get_class_weights(y_train, num_classes)
class_weights_torch = torch.tensor(class_weights, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Class weights: ", class_weights)


model = EmotionResNetAttention(num_features=X.shape[2], seq_len=X.shape[1], num_classes=num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, TP_list, FP_list, TN_list, FN_list, precision_list, recall_list = train_eval_loop(
    model, train_loader, val_loader, test_loader,
    epochs=100, lr=1e-3,
    class_weights=class_weights_torch,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


test_preds = []
test_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y_batch.cpu().numpy())


plot_confusion_matrix(test_labels, test_preds, classes=emotion_labels, filename="test_confusion_matrix.png")
print(classification_report(test_labels, test_preds, target_names=emotion_labels))
plot_training_summary(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist,
                      TP_list, FP_list, TN_list, FN_list,
                      precision_list, recall_list, epoch=len(train_acc_hist))

