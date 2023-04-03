
from lightning.pytorch.cli import LightningCLI

from data.dataloader import SFTPromptDataModule
from lightning_model import LightningPPOModel

def main():

    cli = LightningCLI(LightningPPOModel, SFTPromptDataModule)

if __name__ == "__main__":

    main()
