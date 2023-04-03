
from lightning.pytorch.cli import LightningCLI

from data.dataloader import HHDataModule
from lightning_model import LightningRMModel

def main():

    cli = LightningCLI(LightningRMModel, HHDataModule)

if __name__ == "__main__":

    main()
