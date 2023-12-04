import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, config):
        super(LossWrapper, self).__init__()
        self.config = config
        self.model = model
        self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()

    def forward(self,labels, masks gts, sc_flag):

        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
        else:
            #SC sequence training
            self.model.eval()
            with torch.no_grad():
                # baseline --> sampling greedily.
                #opt.sc_sample_method : "greedy"

                # _ : log probs for each word generated. Upper bound on caption length = 20. shape --> [10, 20, 9488] 
                #calling self
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample',opt={'sample_method': opt.sc_sample_method,'beam_size': opt.sc_beam_size})
            self.model.train()
            #trainable policy 
            #opt.sc_sample_method : "sample"
            # we sample from logits distribution.
            # gen_result --> shape [50,20]. 'opt.train_sample_n' = 5. (5 captions per image generated)

            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, #(N*bsz, 20)
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')


            gts = [gts[_] for _ in gt_indices.tolist()]
            
            # R(c,I) -b : (N policy gen captions * num_images i.e bsz, caption len) 
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt) #(n*B,20)
            # reward is moved to same device as logprobs for each sampled word for each caption.
            reward = torch.from_numpy(reward).to(sample_logprobs)
            # loss = -(R(c,I) -b) * sample_logprobs
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)
            # (R(c,I) -b) averaged over batch. For a given caption, reward is same for log_probs of all the words generated
            print(f"reward --> {reward}")
            out['reward'] = reward[:,0].mean()
            print(f"reward  ---> {out['reward']}")
        out['loss'] = loss
        print(f"loss computed ---> {out['loss']}")
        return out


