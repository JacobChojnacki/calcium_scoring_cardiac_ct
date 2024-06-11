import torch
import wandb

from lightning.pytorch.callbacks import Callback
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose, EnsureType


class LogPredictionsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.post_pred = Compose([EnsureType('tensor'), AsDiscrete(argmax=True, to_onehot=4, num_classes=4)])
        self.post_label = Compose([EnsureType('tensor'), AsDiscrete(to_onehot=4, num_classes=4)])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
        if trainer.current_epoch == 499 or trainer.current_epoch == 999 or trainer.current_epoch == 1599:
            images, labels = batch['images'], batch['labels']

            with torch.no_grad():
                decollated_labels = decollate_batch(labels)
                decollated_outputs = decollate_batch(outputs)

                processed_labels = [self.post_label(label).cpu() for label in decollated_labels]
                processed_preds = [self.post_pred(pred).cpu() for pred in decollated_outputs]

                num_slices = processed_preds[0].shape[-1]
                slice_counts = [(i, torch.sum(processed_labels[0][1, :, :, i] == 1).item()) for i in range(num_slices)]
                top_slices = sorted(slice_counts, key=lambda x: x[1], reverse=True)[:3]
                top_slice_indices = [i for i, _ in top_slices]

                data = []
                for i in top_slice_indices:
                    label_slice = processed_labels[0][1, :, :, i].numpy()
                    masks_labels = {
                        "ground_truth_plaque": {"mask_data": label_slice,
                                                "class_labels": {0: "background", 1: "plaques"}},
                        "ground_truth_aorta": {"mask_data": processed_labels[0][2, :, :, i].numpy() * 2,
                                                "class_labels": {0: "background", 2: "aorta"}},
                        "ground_truth_anatomical_structures": {"mask_data": processed_labels[0][3, :, :, i].numpy() * 3,
                                                                "class_labels": {0: "background", 3: "anatomical_structures"}},
                    }
                    mask_preds = {
                        "prediction_plaques": {"mask_data": processed_preds[0][1, :, :, i].numpy(),
                                            "class_labels": {0: "background", 1: "plaques"}},
                        "prediction_aorta": {"mask_data": processed_preds[0][2, :, :, i].numpy() * 2,
                                            "class_labels": {0: "background", 2: "aorta"}},
                        "prediction_anatomical_structures": {"mask_data": processed_preds[0][3, :, :, i].numpy() * 3,
                                                            "class_labels": {0: "background", 3: "anatomical_structures"}}
                    }

                    image_slice = images.squeeze(0)[:, :, :, i].cpu().numpy()
                    output_slices = outputs.squeeze().cpu().numpy()

                    slice_images = [
                        wandb.Image(image_slice, masks=masks_labels),
                        wandb.Image(output_slices[1, :, :, i]),
                        wandb.Image(output_slices[2, :, :, i]),
                        wandb.Image(output_slices[3, :, :, i]),
                        wandb.Image(image_slice, masks=mask_preds)
                    ]
                    data.append(slice_images)

                    # Free memory for each slice
                    del label_slice, masks_labels, mask_preds, image_slice, output_slices
                    torch.cuda.empty_cache()

                columns = ["images and gt", "activation map plaque", "activation map aorta", "activation map rest", "predictions"]
                table = wandb.Table(data=data, columns=columns)
                wandb.log({f"epoch_{trainer.current_epoch}": table})

                # Free remaining memory
                del decollated_labels, decollated_outputs, processed_labels, processed_preds, data
                torch.cuda.empty_cache()
