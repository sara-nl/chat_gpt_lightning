import math
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup
from typing import Union
from models import GPT, GPTRewardModel, GPTActor, GPTCritic

class Experience():
    def __init__(self, 
                completion: torch.Tensor,
                actor_log_probs: torch.Tensor,
                attention_mask: torch.Tensor,
                kl_penalized_reward: torch.Tensor,
                advantage: torch.Tensor,
                num_actions: int,
                estimated_kl: torch.Tensor,
                values: torch.Tensor,
                action_mask: torch.Tensor):

        self.completion = completion
        self.actor_log_probs = actor_log_probs
        self.attention_mask = attention_mask
        self.kl_penalized_reward = kl_penalized_reward
        self.advantage = advantage
        self.num_actions = num_actions
        self.estimated_kl = estimated_kl
        self.values = values
        self.action_mask = action_mask



class LightningPPOModel(pl.LightningModule):
    def __init__(self,  lora_rank: int=1,
                        lr: float = 6e-4,
                        vocab_size: int=4,
                        n_heads: int = 4,
                        embedding_dim: int = 4,
                        n_layers: int = 128,
                        sequence_length: int = 64,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True,
                        actor_weights: str = "",
                        critic_weights: str = "",
                        policy_eps: float = 0.2,
                        value_eps: float = 0.2,
                        actor_lr: float = 1e-4,
                        critic_lr: float = 1e-4,
                        kl_beta: float = 0.00003,
                        max_new_tokens: int = 1024):

        super().__init__()

        self.actor = GPTActor(lora_rank,
                        vocab_size,
                        n_heads,
                        embedding_dim,
                        n_layers,
                        sequence_length,
                        dropout_rate,
                        activation_checkpointing,
                        use_bias)

        checkpoint_actor = torch.load(actor_weights, map_location="cpu")
        checkpoint_actor = {name.split("model._orig_mod.")[1]: param for name, param in checkpoint_actor["state_dict"].items() if "model._orig_mod." in name}
        self.actor.model.load_state_dict(checkpoint_actor, strict=True)
        
        self.sft_model = GPTActor(lora_rank,
                        vocab_size,
                        n_heads,
                        embedding_dim,
                        n_layers,
                        sequence_length,
                        dropout_rate,
                        activation_checkpointing,
                        use_bias)

        self.sft_model.model.load_state_dict(checkpoint_actor, strict=True)
        del checkpoint_actor

        self.critic = GPTCritic(lora_rank,
                                vocab_size,
                                n_heads,
                                embedding_dim,
                                n_layers,
                                sequence_length,
                                dropout_rate,
                                activation_checkpointing,
                                use_bias)

        checkpoint_critic = torch.load(critic_weights, map_location="cpu")
        checkpoint_critic = {name.split("model._orig_mod.")[1]: param for name, param in checkpoint_critic["state_dict"].items() if "model._orig_mod." in name}
        self.critic.model.backbone.lm_head = nn.Identity()

        self.critic.model.load_state_dict(checkpoint_critic, strict=True)
        

        self.reward_model = GPTRewardModel(lora_rank,
                                        vocab_size,
                                        n_heads,
                                        embedding_dim,
                                        n_layers,
                                        sequence_length,
                                        dropout_rate,
                                        activation_checkpointing,
                                        use_bias)

        self.reward_model.backbone.lm_head = nn.Identity()

        self.reward_model.load_state_dict(checkpoint_critic, strict=True)
        del checkpoint_critic

        self.reward_model.freeze_weights()
        self.critic.model.freeze_weights()
        
        self.sft_model = torch.compile(self.sft_model)
        self.reward_model = torch.compile(self.reward_model)
        self.actor = torch.compile(self.actor)
        self.critic = torch.compile(self.critic)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_eps = policy_eps
        self.value_eps = value_eps
        self.kl_beta = kl_beta
        self.max_new_tokens = max_new_tokens

        #self.automatic_optimization = False
        self.train_actor = True
        

    def forward(self, sequence: torch.Tensor, attention_masks: torch.Tensor):
        return self.model(sequence, attention_masks)

    def value_loss(self, 
                values: torch.Tensor, 
                reward: torch.Tensor, 
                old_values: torch.Tensor):
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L69-L75
        # https://github.com/openai/baselines/issues/91
        values_clipped = old_values + (values - old_values).clamp(-self.value_eps, self.value_eps)
        surrogate_values = torch.max(torch.square(values - reward), torch.square(values_clipped - reward))
        return surrogate_values.mean()  # (B, 1)


    def policy_loss(self, 
                    new_actor_log_probs: torch.Tensor,
                    old_actor_log_probs: torch.Tensor, 
                    advantage: torch.Tensor):

        # reverse the log to get π_new(a_t|s_t) / π_old(a_t|s_t)
        ratio = (new_actor_log_probs -
                 old_actor_log_probs).exp()  # (B, num_actions)
        surrogate_objectives = torch.min(
            ratio * advantage,
            ratio.clamp(1 - self.policy_eps, 1 + self.policy_eps) *
            advantage)  # (B, num_actions)
        # minimize the negative loss -> maximize the objective
        loss = -surrogate_objectives  # (B, num_actions)
        return loss.mean()

    @torch.no_grad()
    def make_experience(self, idx, input_masks, input_lengths):

        self.reward_model.eval()
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()

        # TODO: Batch generate
        completion, attention_mask, num_actions, action_mask = self.actor.batch_generate(idx,
                                                                                         input_masks,
                                                                                         input_lengths,
                                                                                         self.max_new_tokens,
                                                                                         temperature=1.0,
                                                                                         top_k=50)


        actor_log_probs = self.actor.forward_actor(completion,
                                                    attention_mask,  # (B, num_actions)
                                                    num_actions)

        sft_log_probs = self.sft_model.forward_actor(completion, 
                                                    attention_mask, 
                                                    num_actions)  # (B, num_actions)

        values = self.critic.forward_critic(completion,
                                            attention_mask, num_actions).view(-1, 1)  # (B, 1)

        reward = self.reward_model(completion,
                                   attention_mask)  # (B, 1)


        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(reward, 
                                                                    actor_log_probs, 
                                                                    sft_log_probs)

        advantage = kl_penalized_reward - values


        return Experience(completion, 
                        actor_log_probs, 
                        attention_mask,
                        kl_penalized_reward,
                        advantage, 
                        num_actions, 
                        estimated_kl,
                        values, 
                        action_mask)

    def kl_penalized_reward(self,
                            reward: torch.Tensor,
                            log_prob_rl: torch.Tensor,
                            log_prob_sft: torch.Tensor,
                            action_mask: torch.Tensor = None) -> Union[torch.Tensor, torch.Tensor]:

        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio

        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
            
        estimated_kl = estimated_kl.mean(dim=1, keepdim=True)  # estimated_kl -> (B, 1)

        return reward - self.kl_beta * estimated_kl, estimated_kl

    def training_step(self, batch: tuple, batch_idx: int) -> float:
        prompt, input_masks, input_lengths = batch

        max_input_length = torch.max(input_lengths)
        prompt = prompt[:, :max_input_length]
        experience = self.make_experience(prompt, input_masks, input_lengths)

        self.log('KL_loss', experience.estimated_kl.mean(), on_step=True, on_epoch=False)
        self.log('mean_advantage', experience.advantage.mean(),on_step=True, on_epoch=False)
        self.log('mean_reward', experience.kl_penalized_reward.mean(),on_step=True, on_epoch=False)

        # we update either the actor or critic at a time
        # We could manually optimize and do the backwards ourselves
        # but in this way we let lightning handle accelerator, precision and strategy logic
        if self.train_actor:
            self.actor.train()
            curr_actor_log_probs = self.actor.forward_actor(experience.completion, 
                                                            experience.attention_mask, 
                                                            experience.num_actions)

            actor_loss = self.policy_loss(curr_actor_log_probs,
                                        experience.actor_log_probs,
                                        experience.advantage)

            self.log('actor_loss', actor_loss.item(), on_step=True, on_epoch=False)
            self.train_actor = False
            return actor_loss

        else:
            self.critic.train()
            new_values = self.critic.forward_critic(experience.completion, 
                                                    experience.attention_mask,
                                                    experience.num_actions).view(-1, 1)
            
            critic_loss = self.value_loss(new_values, 
                                        experience.kl_penalized_reward, 
                                        experience.values)

            self.log('mean_value', new_values.mean(),on_step=True, on_epoch=False)
            self.log('critic_loss', critic_loss.item(), on_step=True, on_epoch=False)
            self.train_actor = True
            return critic_loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        #actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, betas=(0.9,0.999))
        #critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, betas=(0.9,0.999))
        
        # We can make two specific optimizers but for now 
        # this will do, in this way we let lightning help us more ;)
        optimizer = torch.optim.AdamW([{'params': self.actor.parameters(), 'lr': self.actor_lr},
                        {'params': self.critic.parameters(), 'lr': self.critic_lr}])

        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        #lr_scheduler = get_linear_schedule_with_warmup(
        #                optimizer, num_warmup_steps=15, num_training_steps=100
        #            )

        return optimizer 

class LightningRMModel(pl.LightningModule):
    def __init__(self,  hf_model: str = None, 
                        finetune_method: str = "",
                        lora_rank: int=0,
                        lr: float = 6e-4,
                        vocab_size: int=4,
                        n_heads: int = 4,
                        embedding_dim: int = 4,
                        n_layers: int = 128,
                        sequence_length: int = 64,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True,
                        sft_ckpt_path: str = ""):
        super().__init__()
        
        if hf_model is not None:
            self.model = GPTRewardModel(lora_rank,
                                    vocab_size,
                                    n_heads,
                                    embedding_dim,
                                    n_layers,
                                    sequence_length,
                                    dropout_rate,
                                    activation_checkpointing,
                                    use_bias)

            self.model.backbone = self.model.backbone.from_pretrained(hf_model,
                                                lora_rank,
                                                vocab_size,
                                                n_heads,
                                                embedding_dim,
                                                n_layers,
                                                sequence_length,
                                                dropout_rate,
                                                activation_checkpointing,
                                                use_bias)
            
        else:
            self.model = GPTRewardModel(lora_rank,
                                        vocab_size,
                                        n_heads,
                                        embedding_dim,
                                        n_layers,
                                        sequence_length,
                                        dropout_rate,
                                        activation_checkpointing,
                                        use_bias)

            checkpoint = torch.load(sft_ckpt_path, map_location="cpu")
            # restructure state_dict
            checkpoint = {name.split("model._orig_mod.")[1]: param for name, param in checkpoint["state_dict"].items() if "model._orig_mod." in name}

            print("checkpoint: ", checkpoint)
            self.model.backbone.load_state_dict(checkpoint, strict=True)
        
        # we replace its head for a reward model classifier head
        self.model.backbone.lm_head = nn.Identity()

        # freeze the backbone GPT-sft model
        self.model.freeze_weights()
        
        self.model = torch.compile(self.model)

        self.lr = lr
        self.finetune_method = finetune_method
        

    def forward(self, sequence: torch.Tensor, attention_masks: torch.Tensor):
        return self.model(sequence, attention_masks)

    def KPairWiseLoss(self, scores: torch.Tensor):
        """
        scores: shape of (B, C) where C is number of completions ranked in order
        """
        # Consider scores as [[0.8, 0.7, 0.6]]
        B, C = scores.shape
        # scores = [[[0.8], [0.7], [0.6]]]
        scores = scores[:, :, None]  # (B, C, 1)
        # subtrahend = [[[0.8, 0.8, 0.8],
        #                [0.7, 0.7, 0.7],
        #                [0.6, 0.6, 0.6]]]
        subtrahend = scores.tile((1, C))  # (B, C, C)
        # minuend = [[[0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6]]]
        minuend = subtrahend.transpose(2, 1)  # (B, C, C)
        # diff = [[[0,                 0,                 0],
        #          [log(sigmoid(0.1)), 0,                 0],
        #          [log(sigmoid(0.2)), log(sigmoid(0.1)), 0]]]
        log_odds = torch.tril(torch.log(torch.sigmoid(minuend - subtrahend)),
                              -1)  # (B, C, C)
        total_comparison = math.comb(C, 2)
        expectation = torch.sum(log_odds, dim=(1, 2)) / total_comparison
        loss = -(1 / total_comparison) * expectation.mean()
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> float:
        loss, _, _ = self._step(batch, batch_idx)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        loss, positive_scores, negative_scores = self._step(batch, batch_idx)
        self.log("valid_loss", loss.item(), on_step=True, on_epoch=False)

        true_positives = torch.count_nonzero(positive_scores > negative_scores)
        self.log("batch_val_Accuracy", true_positives / batch[0].shape[0])
        return loss                    

    def _step(self, batch: tuple, batch_idx: int=None, stage: str=None):
        completions, attention_masks = batch
        positive_scores = self(completions[:, 0, :],
                                attention_masks[:, 0, :]) # (B, 1)

        negative_scores = self(completions[:, 1, :],
                                attention_masks[:, 1, :])

        loss = self.KPairWiseLoss(torch.cat([positive_scores, negative_scores], dim=-1))

        return loss, positive_scores, negative_scores

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        #lr_scheduler = get_linear_schedule_with_warmup(
        #                optimizer, num_warmup_steps=15, num_training_steps=100
        #            )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class LightningSFTModel(pl.LightningModule):
    def __init__(self,  hf_model: str = None, 
                        finetune_method: str = "",
                        lora_rank: int=0,
                        lr: float = 6e-4,
                        vocab_size: int=4,
                        n_heads: int = 4,
                        embedding_dim: int = 4,
                        n_layers: int = 128,
                        sequence_length: int = 64,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True):
        super().__init__()
        
        if hf_model is not None:
            self.model = GPT.from_pretrained(hf_model,
                                        lora_rank,
                                        vocab_size,
                                        n_heads,
                                        embedding_dim,
                                        n_layers,
                                        sequence_length,
                                        dropout_rate,
                                        activation_checkpointing,
                                        use_bias)
        else:
            self.model = GPT(lora_rank,
                            vocab_size,
                            n_heads,
                            embedding_dim,
                            n_layers,
                            sequence_length,
                            dropout_rate,
                            activation_checkpointing,
                            use_bias)
        
        self.model = torch.compile(self.model)
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss()

        self.finetune_method = finetune_method
        

    def forward(self, sequence: torch.Tensor):
        return self.model(sequence, )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=False)
        return loss                  

    def _step(self, batch: tuple, batch_idx: int=None, stage: str=None):
        sequence, next_sequence = batch
        outputs = self(sequence)
        loss = self.criterion(outputs.permute(0,2,1), next_sequence)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        #lr_scheduler = get_linear_schedule_with_warmup(
        #                optimizer, num_warmup_steps=15, num_training_steps=100
        #            )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

