# chat_gpt_lightning
A minimal Lightning implementation of chat gpt-2
Largely took over from https://github.com/ethanyanjiali/minChatGPT.

pip install -r requirements.txt

Make sure to paste the best trained model into ./models/SFT/ or ./models/RM/
respectively. 

1. python train_sft.py fit -c config_stf.yml
2. python train_rm.py fit -c config_stf.yml
3. python train_ppo.py fit -c config_stf.yml


