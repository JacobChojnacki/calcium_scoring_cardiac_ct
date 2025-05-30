import lightning as L

from lightning.pytorch.cli import LightningCLI


def main():
    """
    Main function to train the model.
    """
    LightningCLI(
        L.LightningModule,
        L.LightningDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == '__main__':
    main()
