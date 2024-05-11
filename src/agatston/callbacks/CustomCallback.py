import numpy as np
import wandb

from lightning.pytorch.callbacks import Callback
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose, EnsureType
from tqdm import tqdm


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
        post_pred = Compose([EnsureType('tensor'), AsDiscrete(argmax=True, to_onehot=4, num_classes=4)])
        post_label = Compose([EnsureType('tensor'), AsDiscrete(to_onehot=4, num_classes=4)])

        images, labels = batch['images'], batch['labels']
        labels = post_label(decollate_batch(labels)[0]).detach().cpu().numpy()
        pred_plaques = post_pred(decollate_batch(outputs)[0]).detach().cpu().numpy()
        images = images.detach().squeeze(0).cpu().numpy()
        outputs = outputs.squeeze().detach().cpu().numpy()

        num_channels, _, _, num_slices = outputs.shape
        data = []

        with tqdm(total=num_slices) as pbar:
            for i in range(num_slices):
                if np.any(labels[1, :, :, i] == 1):
                    slice_images = [
                        wandb.Image(images[:, :, :, i], masks={
                            "ground_truth_plaque": {"mask_data": labels[1, :, :, i],
                                                    "class_labels": {0: "background", 1: "plaques"}},
                            "ground_truth_aorta": {"mask_data": labels[2, :, :, i] * 2,
                                                   "class_labels": {0: "background", 2: "aorta"}},
                            "groud_truth_anatomical_structures": {"mask_data": labels[3, :, :, i] * 3,
                                                                  "class_labels": {0: "background",
                                                                                   3: "anatomical_structures"}}
                        }),
                        wandb.Image(outputs[1, :, :, i]),
                        wandb.Image(outputs[2, :, :, i]),
                        wandb.Image(outputs[3, :, :, i]),
                        wandb.Image(images[:, :, :, i], masks={
                            "prediction_plaques": {"mask_data": pred_plaques[1, :, :, i],
                                                   "class_labels": {0: "background", 1: "plaques"}},
                            "prediction_aorta": {"mask_data": pred_plaques[2, :, :, i] * 2,
                                                 "class_labels": {0: "background", 2: "aorta"}},
                            "prediction_anatomical_structures": {"mask_data": pred_plaques[3, :, :, i] * 3,
                                                                 "class_labels": {0: "background",
                                                                                  3: "anatomical_structures"}}
                        })
                    ]
                    data.append(slice_images)
                pbar.update(1)

        columns = ["images and gt", "activation map plaque", "activation map aorta",
                   "activation map rest", "predictions"]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({f"epoch_{trainer.current_epoch}": table})
