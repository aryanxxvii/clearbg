import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from clearbg.utils.common import read_yaml, create_directories
from clearbg.constants import PROJECT_ROOT
from clearbg.entity.config_entity import TrainingConfig
from clearbg.utils.utils import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from clearbg.model.u2net import U2NET

class Training:
    def __init__(self, config: TrainingConfig):
        self.config_manager = config
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()
        self.training_config = self.config_manager.get_training_config()

        self.model_name = 'u2net'
        self.model_dir = PROJECT_ROOT / self.training_config.root_dir / 'saved_models' / self.model_name
        create_directories([self.model_dir])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch_num = self.training_config.epochs
        self.batch_size_train = self.training_config.batch_size
        self.save_frq = 2000  # Save the model every 2000 iterations
        self.bce_loss = nn.BCELoss(reduction='mean')

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print(f"l0: {loss0.data.item():.3f}, l1: {loss1.data.item():.3f}, l2: {loss2.data.item():.3f}, "
              f"l3: {loss3.data.item():.3f}, l4: {loss4.data.item():.3f}, l5: {loss5.data.item():.3f}, l6: {loss6.data.item():.3f}\n")
        
        return loss0, total_loss

    def prepare_data(self):
        data_dir = self.data_ingestion_config.root_dir
        tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
        tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)
        image_ext = '.jpg'
        label_ext = '.png'

        tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*' + image_ext))

        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]
            imidx = ".".join(img_name.split(".")[:-1])
            tra_lbl_name_list.append(os.path.join(data_dir, tra_label_dir, imidx + label_ext))

        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")

        train_num = len(tra_img_name_list)

        salobj_dataset = SalObjDataset(
            img_name_list=tra_img_name_list,
            lbl_name_list=tra_lbl_name_list,
            transform=transforms.Compose([RescaleT(self.training_config.image_size),
                                          RandomCrop(288),
                                          ToTensorLab(flag=0)]))

        salobj_dataloader = DataLoader(salobj_dataset, batch_size=self.batch_size_train, shuffle=True, num_workers=1)

        return salobj_dataloader, train_num

    def initialize_model(self):
        net = U2NET(3, 1)
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.training_config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return net, optimizer

    def train(self):
        salobj_dataloader, train_num = self.prepare_data()
        net, optimizer = self.initialize_model()
        
        print("---start training...")

        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

        for epoch in range(self.epoch_num):
            net.train()

            for i, data in enumerate(salobj_dataloader):
                ite_num += 1
                ite_num4val += 1

                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                d0, d1, d2, d3, d4, d5, d6 = net(inputs)
                loss2, loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

                # Backward + optimize
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()

                print(f"[epoch: {epoch+1}/{self.epoch_num}, batch: {(i+1)*self.batch_size_train}/{train_num}, "
                      f"ite: {ite_num}] train loss: {running_loss / ite_num4val:.3f}, tar: {running_tar_loss / ite_num4val:.3f}")

                # Save model periodically
                if ite_num % self.save_frq == 0:
                    model_save_path = os.path.join(self.model_dir, f"{self.model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth")
                    torch.save(net.state_dict(), model_save_path)
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    ite_num4val = 0
                    net.train()

# Instantiate and start training
if __name__ == "__main__":
    trainer = Training()
    trainer.train()
