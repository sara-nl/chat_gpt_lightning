# chat_gpt_lightning
A minimal Lightning implementation of chat gpt-2.
Largely from https://github.com/ethanyanjiali/minChatGPT but minus fat.

pip install -r requirements.txt

Make sure to paste the best trained model into ./models/SFT/ or ./models/RM/
respectively. 

```
cd ./data/
python prepare_sft_data.py
```
 
Train supervised finetuning:
 
```
python train_sft.py fit -c config_stf.yml
```

Train reward model:

```
python train_rm.py fit -c config_rm.yml
```

Train PPO:

```
python train_ppo.py fit -c config_ppo.yml
```

TODO:

1. Train a small model gpt-2
2. Train on multi node/multi gpu?
2. Add LLaMA
