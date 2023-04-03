
from lightning.pytorch.cli import LightningCLI

from data.dataloader import SFTDataModule
from lightning_model import LightningSFTModel

def main():

    cli = LightningCLI(LightningSFTModel, SFTDataModule)

if __name__ == "__main__":

    main()
