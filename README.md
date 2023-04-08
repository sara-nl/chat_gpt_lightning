# chat_gpt_lightning
A minimal Lightning implementation of chat gpt-2.
Largely from https://github.com/ethanyanjiali/minChatGPT but minus fat.

pip install -r requirements.txt


```
cd ./data/anthropic/
python prepare_sft_data.py
```
 
Train supervised finetuning:
 
```
python train_sft.py fit -c ./configs/config_sft.yml
```

Make sure to paste the best trained sft model into ./models/SFT/

Train reward model:

```
python train_rm.py fit -c ./configs/config_rm.yml
```

Add the best reward model to ./models/RM/

Train PPO:

```
python train_ppo.py fit -c ./configs/config_ppo.yml
```

TODO:

1. Train a small model gpt-2, DONE
2. Train big model gpt-2
3. train with lora.
4. Train on multi node/multi gpu? DONE
5. Add LLaMA, meh skip this one, better use pythia
