import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from medsam.build_sam import sam_model_registry
from models.contour_predictor import ContourPredictor
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
join = os.path.join


class Solver(object):
    def __init__(self, args, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = None
        self.optimizer = None
        self.img_ch = 3
        self.output_ch = 1

        self.mseloss = nn.MSELoss()

        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.num_epochs_decay = args.num_epochs_decay
        self.model_type = args.model_type
        self.model_path = None
        self.result_path = None
        self.mode = args.mode
        self.device = torch.device('cuda')
        self.continue_train = args.continue_train
        self.checkpoint = args.checkpoint
        self.resume = args.resume
        self.work_dir = args.work_dir
        self.task_name = args.task_name
        self.build_model()

    def build_model(self):
        sam_model = sam_model_registry[self.model_type](checkpoint='./medsam/medsam_vit_b.pth')
        sam_model.requires_grad_(False)
        self.model = ContourPredictor(image_encoder=sam_model.image_encoder, contour_num=1)
        params = list(self.model.image_encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))
        self.model.to(self.device)

    def train(self):
        self.model.train()
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = join(self.work_dir, self.task_name + "-" + run_id)
        os.makedirs(model_save_path, exist_ok=True)
        losses = []
        best_loss = 1e10
        start_epoch = 0

        if self.resume is not None and self.resume != '':
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume, map_location=self.device)
                start_epoch = checkpoint["epoch"] + 1
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.continue_train:
            self.model.load_state_dict(torch.load(self.checkpoint)['model'])

        lr = self.lr

        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0
            for step, (image, gt) in enumerate(tqdm(self.train_dataloader)):
                image, gt = image.to(self.device), gt.to(self.device)
                curve_pred = self.model(image)

                gt = gt.squeeze(1)
                curve_pred = curve_pred.squeeze(1)

                loss = self.mseloss(curve_pred, gt)

                epoch_loss += loss.item()
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss /= (step + 1)
            losses.append(epoch_loss)

            print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, join(model_save_path, "curvedetect_model_latest.pth"))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "curvedetect_model_best.pth"))
                torch.save(checkpoint, join(self.work_dir, 'curvedetect_model_best.pth'))

            plt.plot(losses)
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(model_save_path, self.task_name + "_train_loss.png"))
            plt.close()

    def test(self):
        model_path = './work_dir/curvedetect_model_best.pth'
        self.model.load_state_dict(torch.load(model_path)['model'])
        self.model.eval()
        index = 0

        for _, (images, GT) in enumerate(self.test_dataloader):
            images = images.to(self.device)
            GT = GT.to(self.device)

            for i in range(len(images)):
                image = images.detach().cpu().numpy()[i][0]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                image_img = Image.fromarray((image * 255).astype(np.uint8))
                image_img.show()

                gt = GT.detach().cpu().numpy()[i][0]
                gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
                gt_img = Image.fromarray((gt * 255).astype(np.uint8))
                gt_img.show()

                res_ori = self.model(images).detach().cpu().numpy()

                for j in range(res_ori.shape[1]):
                    res = res_ori[i][j]
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    res_img = Image.fromarray((res * 255).astype(np.uint8))
                    res_img.show()

                index += 1
                input('Press Enter to Continue...')