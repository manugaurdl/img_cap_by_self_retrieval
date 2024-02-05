from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, LogitsProcessorList, LogitsProcessor
import torch

class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, do_sample):
        self.eos_token_id = tokenizer.eos_token_id #50256

        # f : tokenizer.convert_ids_to_tokens --> decodes token  i.e f(500) = "walk" 
        # There are 121 strings that contain "." --> all of them are treated as stop tokens
        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # iterate each batch of prefix tokens ; input_ids (B , 10)
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min
                scores[i, self.vocab_size :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
        return scores


def maybe_patch_gpt(self, max_embeddings : int):
    if not getattr(self.gpt, "_patched", False):
        self.gpt._patched = True
        #increase gpt nn.embedding from vocab size --> + 5000
        self.gpt.resize_token_embeddings(len(self.tokenizer) + max_embeddings)
        
        # bias for output embeddings (768) added (55257)
        # its requires grad is False?
        if self.gpt.get_output_embeddings().bias is None:
            self.gpt.get_output_embeddings().bias = torch.nn.Parameter(
                torch.tensor([0.0] * (len(self.tokenizer) + max_embeddings))
            )
            self.gpt.get_output_embeddings().bias.requires_grad = False
            self.gpt.get_output_embeddings().to(
                self.gpt.get_output_embeddings().weight.device
            )
            self.gpt._originally_with_no_bias = True
        else:
            self.gpt._originally_with_no_bias = False
        self.gpt.get_output_embeddings().bias.data[-max_embeddings:] = float("-inf")
        print(f"Patched GPT : Increased vocab by {max_embeddings}")


def hf_sample(token_emb, model, config):
    """
    input : (B , num_prefix_tokens, 768)
    """
    prompts = token_emb
    bsz, prefix_len, h_dim = prompts.shape
    prompts_flat = prompts.reshape(-1, prompts.size(-1))
    ### At each batch --> we have a different look up table for GPT. The last bsz * 10 tokens correspond to prefix. 
    start = len(model.tokenizer)  # 50257
    tokens_per_batch = prompts_flat.shape[0] #bsz * prefix_len
    end = start + tokens_per_batch

    input_ids = torch.arange(start, end).view(*prompts.shape[:2]).to(prompts.device) # Adding prefix tokens to GPT vocabulary.
    model.gpt.get_input_embeddings().weight.data[start:end] = prompts_flat # Add prefix tokens's embeddings to GPT look up table
    generated = model.gpt.generate(
                input_ids,
                do_sample=False,
                max_length= config['max_length'],
                num_beams=5,
                num_return_sequences=1,
                logits_processor=LogitsProcessorList([model.logits_processor]),
                top_k=len(model.tokenizer),
            )
    indices = generated[:, prefix_len:]
    decoded_cap = [i.split("<|endoftext|>")[0] for i in model.tokenizer.batch_decode(indices)]

    return decoded_cap