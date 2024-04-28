import lightning
import lightning as L

from fire import Fire
from lightning.pytorch.cli import LightningCLI


def main():
    """
    Main function to train the model.
    """
    L.seed_everything(42, workers=True)
    LightningCLI(
        L.LightningModule,
        L.LightningDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    Fire(main)
