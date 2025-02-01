from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import torch
import os

TRAIN_LOSSES = 'train_losses'
VAL_LOSSES = 'val_losses'


class ModelUtils:
    @classmethod
    def save_model(cls, model, full_model_save_path, model_name=None) -> dict:
        if not os.path.exists(full_model_save_path):
            os.makedirs(full_model_save_path)

        full_save_path = os.path.join(full_model_save_path, f'{model_name}.pt' if model_name else 'best_model.pt')
        print(f'Best model will be saved to {full_save_path}.')
        torch.save(obj=model.state_dict(), f=full_save_path)
        return full_save_path

    @classmethod
    def save_training_data(cls, save_path, epoch, data):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        log_data = []
        log_data_path = os.path.join(save_path, 'train_logs.txt')

        if os.path.exists(log_data_path):
            with open(log_data_path, 'r') as file:
                log_data = [line.strip() for line in file.readlines()]
        else:
            log_data.append(f'Train Length: {data["train_len"]}')
            log_data.append(f'Batch Size: {data["batch"]}')
            log_data.append(f'Epochs: {data["epochs"]}')
            log_data.append(f'Learning rate: {data["lr"]}')
            log_data.append('Training Logs: ------------------------------')
        log_data.append(f'e={epoch}; training_loss={data["train_loss"]}; validation_loss={data["val_loss"]}')

        with open(log_data_path, 'w') as file:
            for item in log_data:
                file.write(str(item) + '\n')


class DatasetUtils:
    def __init__(self, data_path, resize=None, data_type='tensor', split_per=None, translate=None,
                 rand_resize_scale=None, blur_size=None):
        self.data_path = data_path
        self.resize = resize
        self.data_type = data_type
        self.split_per = split_per
        self.translate = translate
        self.rand_resize_scale = rand_resize_scale
        self.blur_size = blur_size

    def create_dataset(self):
        transform = self.create_transform()
        data = ImageFolder(root=self.data_path, transform=transform)
        if self.translate is not None or self.rand_resize_scale is not None or self.blur_size is not None:
            print('AUGMENTATION ENABLED')
            transform = self.create_transform(augment=True)
            augmented_data = ImageFolder(root=self.data_path, transform=transform)
            data = torch.utils.data.ConcatDataset([data, augmented_data])
        if self.split_per:
            train_ds, test_ds = torch.utils.data.random_split(data, self.split_per)
            print(f'DATASET CREATED: split1={len(train_ds)} | split1={len(test_ds)}')
            return train_ds, test_ds
        print(f'DATASET CREATED: data={len(data)}')
        return data

    def create_transform(self, augment=False):
        transforms_params = []
        if self.resize:
            transforms_params.append(transforms.Resize(self.resize))
        if self.data_type == 'tensor':
            transforms_params.append(transforms.ToTensor())
        if augment:
            if self.translate:
                transforms_params.append(transforms.RandomAffine(translate=self.translate, degrees=0))
            if self.rand_resize_scale:
                transforms_params.append(transforms.RandomResizedCrop(size=self.resize, scale=self.rand_resize_scale))
            if self.blur_size:
                transforms_params.append(transforms.GaussianBlur(kernel_size=self.blur_size))

        transform = transforms.Compose(transforms_params)
        return transform


class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AnomalyDetector:
    def __init__(self, data_path, epochs, batch_size=16, lr=0.001, shuffle=True, resize=None, data_type='tensor',
                 split_per=None, model_name=None, model_path='model', patience=5, save_path='history', translate=None,
                 rand_resize_scale=None, blur_size=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = data_path
        self.resize = resize
        self.data_type = data_type
        self.split_per = split_per if split_per else [0.8, 0.2]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.learning_rate = lr
        self.model_name = model_name
        self.model_path = model_path
        self.model = AutoEncoderModel()
        self.criterion = nn.MSELoss()
        self.train_ds, self.val_ds = None, None
        self.patience = patience
        self.save_path = save_path
        self.translate = translate
        self.rand_resize_scale = rand_resize_scale
        self.blur_size = blur_size

    # TRAIN
    def train(self):
        # Load Data
        ds_utils = DatasetUtils(data_path=self.data_path, resize=self.resize, data_type=self.data_type,
                                split_per=self.split_per, translate=self.translate, rand_resize_scale=self.rand_resize_scale,
                                blur_size=self.blur_size)
        self.train_ds, self.val_ds = ds_utils.create_dataset()
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=self.shuffle)

        # Load Model
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        # Train Model
        self.model.train()
        losses = {TRAIN_LOSSES: [], VAL_LOSSES: []}
        prev_loss = 100000
        for epoch in range(self.epochs):
            losses, optimizer = self._train_epoch(data_loader=train_loader, epoch=epoch, optimizer=optimizer,
                                                  val_loader=val_loader, losses=losses)
            data = {
                "train_len" : len(self.train_ds),
                "batch": self.batch_size,
                "epochs": self.epochs,
                "lr": self.learning_rate,
                "train_loss": float(losses[TRAIN_LOSSES][-1]),
                "val_loss": float(losses[VAL_LOSSES][-1])
            }

            val_loss = losses[VAL_LOSSES][-1]
            scheduler.step(val_loss)

            if val_loss < prev_loss:
                ModelUtils.save_model(model=self.model,
                                      full_model_save_path=self.model_path,
                                      model_name=self.model_name)
                prev_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                break

            ModelUtils.save_training_data(save_path=self.save_path, epoch=epoch, data=data)

    def _train_epoch(self, data_loader, epoch, optimizer, val_loader, losses):
        running_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as tbar:
            for batch in tbar:
                running_loss, optimizer = self.__training_loop(batch=batch, optimizer=optimizer,
                                                               running_loss=running_loss)
            avg_loss = running_loss / len(data_loader)
            losses[TRAIN_LOSSES].append(round(avg_loss, 6))

        losses = self.__validate(data_loader=val_loader, avg_loss=avg_loss, losses=losses, epoch=epoch)
        return losses, optimizer

    def __training_loop(self, batch, optimizer, running_loss):
        images, _ = batch
        images = images.to(self.device)
        recon = self.model(x=images)

        # Backpropagation
        optimizer.zero_grad()
        loss = self.criterion(recon, images)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        return running_loss, optimizer

    def __validate(self, data_loader, avg_loss, losses, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(data_loader, desc=f"Validating:", unit="batch") as tbar:
                for batch in tbar:
                    val_loss = self.__validation_loop(batch, val_loss)
            avg_val_loss = val_loss / len(data_loader)
            losses[VAL_LOSSES].append(round(avg_val_loss, 6))
        self.model.train()
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, Val Loss: {avg_val_loss}")
        return losses

    def __validation_loop(self, batch, val_loss):
        images, _ = batch
        images = images.to(self.device)
        recon = self.model(x=images)
        loss = self.criterion(recon, images)
        val_loss += loss.item()
        return val_loss

    # TEST
    def test(self, data, model_path=None, threshold=None):
        print('AnomalyDetector | TESTING...')
        if model_path:
            self.model.load_state_dict(torch.load(f=model_path))
            print(f'Model loaded from {model_path}')
            self.model.to(self.device)

        self.model.eval()
        threshold = threshold if threshold else self._get_generation_error_threshold()
        with torch.no_grad():
            test_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=self.shuffle)
            mse_per_image, y_pred_per_img, y_true_per_img = [], [], []
            with tqdm(test_loader, desc=f"Testing model {model_path}", unit="batch") as tbar:
                for batch in tbar:
                    images, labels = batch
                    images = images.to(self.device)
                    recon = self.model(x=images)

                    for i in range(images.shape[0]):
                        mse = F.mse_loss(recon[i], images[i], reduction='mean')
                        mse_per_image.append(mse.item())
                        y_pred = 1 * (mse >= threshold)
                        y_pred_per_img.append(y_pred.cpu())
                        y_true = 0 if labels[i] == data.classes.index("good") else 1
                        y_true_per_img.append(y_true)

                        # print(y_pred.cpu(), y_true, 'TRUE' if y_pred == y_true else '')
            result = [x == y for x, y in zip(y_pred_per_img, y_true_per_img)]
            anomaly_score = [a == 1 and b == 1 for a, b in zip(y_pred_per_img, y_true_per_img)]
            good_score = [a == 0 and b == 0 for a, b in zip(y_pred_per_img, y_true_per_img)]
            # print(result)
            print(f'Model accuracy is: {result.count(True)} / {len(y_pred_per_img)}')
            print(f'Anomaly accuracy is: {anomaly_score.count(True)} / {y_true_per_img.count(1)}')
            print(f'Good accuracy is: {good_score.count(True)} / {y_true_per_img.count(0)}')

    def _get_generation_error_threshold(self, data=None):
        if not data:
            if self.train_ds:
                data = self.train_ds
            else:
                ds_utils = DatasetUtils(data_path=self.data_path, resize=self.resize, data_type=self.data_type)
                data = ds_utils.create_dataset()

        with torch.no_grad():
            data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=self.shuffle)
            errors = []
            with tqdm(data_loader, desc=f"Calculating anomaly threshold...", unit="batch") as tbar:
                for batch in tbar:
                    images, _ = batch
                    images = images.to(self.device)
                    recon = self.model(images)
                    for i in range(images.shape[0]):
                        mse = F.mse_loss(recon[i], images[i], reduction='mean')
                        errors.append(mse.item())
        recon_error = np.array(errors)
        threshold = np.mean(recon_error) + 3 * np.std(recon_error)
        print(f'Anomaly threshold: {threshold}')
        return threshold
