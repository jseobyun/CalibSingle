import os
import cv2
import time
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from filter.vae_model import OutlierVAE, View




def binarize(img):
    # Otsu's thresholding after Gaussian filtering
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_out = np.float32(img_out) / 255.0
    return img_out

def load_crop(img_dir, num_max=None):
    # load corner crops
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
    if num_max is not None:
        img_paths = img_paths[:num_max]
    crops = []
    for img_path in img_paths:
        crop = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        crop = binarize(crop)
        crop_h, crop_w = np.shape(crop)[:2]
        crops.append(crop.reshape(-1, 1, crop_h, crop_w))
    crops = np.concatenate(crops, axis=0)
    crops = torch.from_numpy(crops)
    return crops

def load_crop_np(img_dir, num_max=None):
    # load corner crops
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
    if num_max is not None:
        img_paths = img_paths[:num_max]
    crops = []
    for img_path in img_paths:
        crop = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        crop_h, crop_w = np.shape(crop)[:2]
        crops.append(crop.reshape(-1, crop_h, crop_w))
    crops = np.concatenate(crops, axis=0)
    return crops

def train(img_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    batch_size = 1000
    device = "cuda"
    lr = 0.001
    n_epochs = 500
    kl_weight = 0.01

    # load corner crops
    crops = load_crop(img_dir)

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # truncate batch
    n_batch = len(crops) // batch_size
    crops_trunc = crops[0:batch_size * n_batch]
    print("Train input: {} crops | batch_size={} | n_batch={} | trunc={} | unused={}"
          .format(len(crops), batch_size, n_batch, len(crops_trunc), len(crops) - len(crops_trunc)))

    # initialize model
    model = OutlierVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # save losses for monitoring
    losses = {"total": [], "kld": [], "recon": []}

    for e in tqdm(range(n_epochs)):
        # shuffle
        indices = np.random.shuffle(np.arange(0, len(crops_trunc)))
        crops_trunc_shuffled = crops_trunc[indices].squeeze()

        # losses in current epoch
        losses_curr = {"total": 0.0, "kld": 0.0, "recon": 0.0}

        for batch_idx in range(n_batch):
            in_batch = crops_trunc_shuffled[batch_idx * batch_size:(batch_idx + 1) * batch_size].unsqueeze(1).to(device)

            # forward
            recons, mu, logvar = model(in_batch, is_training=True)
            l_kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)  # [batch_size]
            l_recons = torch.sum(torch.sum(((in_batch - recons) ** 2).squeeze(), dim=1), dim=1)  # [batch_size]
            loss = torch.mean(l_recons + kl_weight * l_kld)

            l_kld = torch.mean(l_kld)
            l_recons = torch.mean(l_recons)

            losses_curr["total"] += loss.item() / n_batch
            losses_curr["kld"] += l_kld.item() / n_batch
            losses_curr["recon"] += l_recons.item() / n_batch

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in losses_curr.items():
            losses[k].append(v)

        if len(losses["total"]) > 0:
            tqdm.write(
                'Epoch {}/{} | loss: initial={:.4f} curr={:.4f} | recon loss={:.4f}, kld loss={:.4f}'
                    .format(e + 1, n_epochs,  losses["total"][0], losses["total"][-1], losses["recon"][-1],
                    losses["kld"][-1])
            )

        if (e > 0 and (e + 1) % 20 == 0) or e == n_epochs - 1:
            # save model
            model_save_path = os.path.join(save_dir, "vae_model.pt")
            torch.save(model, model_save_path)
            print("\nModel saved: {}".format(model_save_path))

            plt.figure()
            plt.plot(losses["total"], linewidth=3, label="total")
            plt.plot(losses["recon"], linewidth=1, label="recon")
            plt.plot(losses["kld"], linewidth=1, label="kld")

            plt.legend(loc="upper right")
            plt.title("Train loss (epoch={})\nfinal loss={:.4f} | recon={:.4f}, kld={:.4f}".format(e + 1, losses["total"][-1],
                                                                                             losses["recon"][-1],
                                                                                             losses["kld"][-1]))
            plt.grid()
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.xlim([0, n_epochs])

            save_path = os.path.join(save_dir, "train_loss_plot.png")
            plt.savefig(save_path, dpi=150)
            print("Train plot saved: {}".format(save_path))
            plt.close()

            if e == n_epochs - 1:
                print("Train plot saved: {}".format(save_path))

    model_save_path = os.path.join(save_dir, "vae_model.pt")
    torch.save(model, model_save_path)
    print("Model saved: {}".format(model_save_path))


def inference(img_dir, save_dir, model_path, save_result_imgs=False):
    device = "cuda"
    # load model
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    # save forward run reconstruction losses
    output_path = os.path.join(save_dir, "vae_forward_result.json")
    result = {}

    crops = load_crop(img_dir)

    # for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)



    losses = {"recon": []}
    pbar = tqdm(total=len(crops))

    if save_result_imgs:
        outputs = []
    for i in range(len(crops)):
        input_crops = crops[i].unsqueeze(0).to(device)

        # forward
        recons, _, _ = model(input_crops, is_training=False)

        # losses
        # l_kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        l_recons = float(torch.sum(((input_crops - recons) ** 2)).detach().squeeze().cpu())

        losses["recon"].append(l_recons)

        # save losses
        result[i] = l_recons

        if save_result_imgs:
            outputs.append({"idx": i, "input": input_crops.squeeze().detach().cpu().numpy(),
                            "recon": recons.squeeze().detach().cpu().numpy(), "loss": l_recons})

        pbar.update(1)
    pbar.close()

    with open(output_path, 'w+') as f:
        json.dump(result, f, indent=4)

    print("VAE forward result saved to: {}".format(output_path))

    if save_result_imgs:
        n_crops = len(outputs)
        n_cols = 10
        n_rows_max = 10

        n_rows, rem = divmod(n_crops, n_cols)
        n_plots, rem2 = divmod(n_rows * 2, n_rows_max)

        if rem2 > 0:
            n_plots += 1
        if rem > 0:
            n_rows += 1

        result_dir = os.path.join(save_dir, "forward_result")
        os.makedirs(result_dir, exist_ok=True)

        crop_idx = 0
        for plot_idx in range(n_plots):
            result_path = os.path.join(result_dir, "forward_{}.png".format(plot_idx))
            s = 2
            fig, ax = plt.subplots(nrows=n_rows_max, ncols=n_cols, squeeze=True,
                                   figsize=(s * n_cols, 1.2 * s * n_rows_max))
            ax = ax.ravel()
            for r in range(0, n_rows_max, 2):
                for c in range(n_cols):
                    a0 = ax[r * n_cols + c]
                    a1 = ax[(r + 1) * n_cols + c]
                    if crop_idx < len(crops):
                        output = outputs[crop_idx]
                        a0.imshow(output["input"], cmap="gray")
                        a0.set_title(
                            "({}/{})\n{} | {:.2f}".format(crop_idx + 1, n_crops, output["idx"], output["loss"]))
                        a0.set_xticks([])
                        a0.set_yticks([])

                        a1.imshow(output["recon"], cmap="gray")
                        a1.set_title("({}/{})\nreconstructed".format(crop_idx + 1, n_crops))
                        a1.set_xticks([])
                        a1.set_yticks([])

                        crop_idx += 1
                    else:
                        a0.axis(False)
                        a1.axis(False)
            fig.subplots_adjust(top=0.95)
            plt.suptitle("[{}/{}] {}/{} forwards".format(plot_idx + 1, n_plots, (plot_idx + 1) * n_cols * n_rows_max,
                                                         len(crops)))
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()
            print("Forward plot saved [{}/{}]: {}".format(plot_idx + 1, n_plots, result_path))


def determine_outliers(img_dir, save_dir, model_path, outlier_thres_ratio=0.001, save_imgs=False):
    # load crops
    crops = load_crop(img_dir)
    crops_gray = load_crop_np(img_dir)
    # recon loss from forward vae
    with open(os.path.join(save_dir, "vae_forward_result.json"), "r") as f:
        vae_result = json.load(f)

    corner_indices = np.array(list(vae_result.keys()))
    recon_losses = []
    for i in corner_indices:
        recon_losses.append(vae_result[i])
    recon_losses = np.array(recon_losses)

    idx = np.argsort(recon_losses)[::-1]
    sorted_losses = recon_losses[idx]
    sorted_indices = corner_indices[idx]

    n_items = len(corner_indices)
    outliers_indices = sorted_indices[0:int(n_items * outlier_thres_ratio)]
    outliers_losses = sorted_losses[0:int(n_items * outlier_thres_ratio)]

    outliers = {}
    for i in range(len(outliers_indices)):
        crop_idx = outliers_indices[i]
        loss = outliers_losses[i]
        outliers[crop_idx] = {"crop_idx": int(crop_idx), "recon_loss": loss}

    if save_imgs:
        # load vae model to run forward passes
        # load model
        model = torch.load(model_path)
        model.eval()

        result_dir = os.path.join(save_dir, "outlier_vis")
        os.makedirs(result_dir, exist_ok=True)

        # load corner images
        crops_for_plot = []
        crops_recon = []
        pbar = tqdm(total=len(outliers.keys()))
        for _, md in outliers.items():
            pbar.update(1)

            crop_idx = int(md["crop_idx"])
            crop_binary = crops[crop_idx]
            crop_gray = crops_gray[crop_idx,:,:]

            # forward
            input_crop = crop_binary.unsqueeze(0).to("cuda")
            with torch.no_grad():
                recon, _, _ = model(input_crop, is_training=False)
            recon = recon.detach().squeeze().cpu().numpy()

            crops_recon.append(recon)
            crops_for_plot.append(crop_gray)
        pbar.close()

        n_crops = len(crops_for_plot)
        n_cols = 10
        n_rows_max = 10

        n_rows, rem = divmod(n_crops, n_cols)
        n_plots, rem2 = divmod(n_rows * 2, n_rows_max)

        if rem2 > 0:
            n_plots += 1
        if rem > 0:
            n_rows += 1

        i = 0
        for plot_idx in range(n_plots):
            result_dir = os.path.join(save_dir, "outlier_vis")
            save_path = os.path.join(result_dir, "outliers_{}.png".format(plot_idx))
            s = 2
            fig, ax = plt.subplots(nrows=n_rows_max, ncols=n_cols, squeeze=True,
                                   figsize=(s * n_cols, 1.2 * s * n_rows_max))
            ax = ax.ravel()
            for r in range(0, n_rows_max, 2):
                for c in range(n_cols):
                    a0 = ax[r * n_cols + c]
                    a1 = ax[(r + 1) * n_cols + c]
                    if i < len(crops_for_plot):
                        key = list(outliers.keys())[i]
                        md = outliers[key]
                        a0.imshow(crops_for_plot[i], cmap="gray")
                        a0.set_title("({}/{})\n{} | {:.2f}".format(i + 1, n_crops, md["crop_idx"], md["recon_loss"]))

                        a0.set_xticks([])
                        a0.set_yticks([])

                        a1.imshow(crops_recon[i], cmap="gray")
                        a1.set_title("({}/{})\nreconstructed".format(i + 1, n_crops))
                        a1.set_xticks([])
                        a1.set_yticks([])

                        i += 1
                    else:
                        a0.axis(False)
                        a1.axis(False)
            fig.subplots_adjust(top=0.95)
            plt.suptitle("[{}/{}] {}/{} outlier corners (total={})".format(plot_idx + 1, n_plots,
                                                                           (plot_idx + 1) * n_cols * n_rows_max,
                                                                           len(crops_for_plot), n_items))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print("Outlier plot saved [{}/{}]: {}".format(plot_idx + 1, n_plots, save_path))

if __name__ == "__main__":
    #train("/home/jseob/Desktop/yjs/corner_patches", "/home/jseob/Desktop/yjs/outlier_detector")
    inference("/home/jseob/Desktop/yjs/corner_patches",
              "/home/jseob/Desktop/yjs/outlier_detector",
              "./vae_model.pt",
              False)
    # determine_outliers("/home/jseob/Desktop/yjs/corner_patches",
    #           "/home/jseob/Desktop/yjs/outlier_detector",
    #           "/home/jseob/Desktop/yjs/outlier_detector/vae_model.pt",
    #           0.001,
    #           True)