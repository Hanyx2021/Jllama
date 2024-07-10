# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from tqdm import tqdm
import jittor as jt

# jt.flags.use_cuda = 1

from Jllama.model import ModelArgs, Transformer
from Jllama.tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        start_time = time.time()
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        model = Transformer(model_args)
        print("Start loading model")
        model.load("Meta-Llama-3-8B/consolidated.00.pkl")
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    with jt.no_grad():
        def generate(
            self,
            prompt_tokens: List[List[int]],
            max_gen_len: int,
            temperature: float = 0.6,
            top_p: float = 0.9,
            logprobs: bool = False,
            echo: bool = False,
        ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
            """
            Generate text sequences based on provided prompts using the language generation model.

            Args:
                prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
                max_gen_len (int): Maximum length of the generated text sequence.
                temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
                top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
                logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
                echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

            Returns:
                Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

            Note:
                This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
                If logprobs is True, token log probabilities are computed for each generated token.

            """
            params = self.model.params
            bsz = len(prompt_tokens)
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

            min_prompt_len = min(len(t) for t in prompt_tokens)
            max_prompt_len = max(len(t) for t in prompt_tokens)
            assert max_prompt_len <= params.max_seq_len
            total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

            pad_id = self.tokenizer.pad_id
            tokens = jt.full((bsz, total_len), pad_id,dtype=jt.int32)
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = jt.Var(t)
            if logprobs:
                token_logprobs = jt.zeros_like(tokens,dtype=jt.float32)

            prev_pos = 0
            eos_reached = jt.Var([False] * bsz)
            input_text_mask = tokens != pad_id

            if min_prompt_len == total_len:
                logits = self.model.execute(tokens, prev_pos)
                token_logprobs = -jt.nn.cross_entropy(
                    output = logits.transpose(1, 2),
                    target = tokens,
                    ignore_index = pad_id,
                    reduction="none",
                )
            stop_tokens = jt.Var(list(self.tokenizer.stop_tokens))
            for cur_pos in tqdm(range(min_prompt_len, total_len)):
                logits = self.model.execute(tokens[:, prev_pos:cur_pos], prev_pos)
                if temperature > 0:
                    probs = jt.nn.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = jt.argmax(logits[:, -1], dim=-1)
                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated

                next_token = jt.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -jt.nn.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                eos_reached = eos_reached.logical_or(input_text_mask[:, cur_pos].logical_not().logical_and(jt.isin(next_token, stop_tokens)))
                prev_pos = cur_pos
                if all(eos_reached):
                    break
            if logprobs:
                token_logprobs = token_logprobs.tolist()
            out_tokens, out_logprobs = [], []
            for i, toks in enumerate(tokens.tolist()):
                # cut to max gen len
                start = 0 if echo else len(prompt_tokens[i])
                toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
                probs = None
                if logprobs:
                    probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
                # cut to after eos tok if any
                for stop_token in self.tokenizer.stop_tokens:
                    try:
                        eos_idx = toks.index(stop_token)
                        toks = toks[:eos_idx]
                        probs = probs[:eos_idx] if logprobs else None
                    except ValueError:
                        pass
                out_tokens.append(toks)
                out_logprobs.append(probs)
            return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = jt.sort(probs, dim=-1, descending=True)
    probs_sum = jt.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.divide(probs_sort.sum(dim=-1, keepdim=True))
    next_token = jt.multinomial(probs_sort, num_samples=1, replacement = True)
    next_token = jt.gather(probs_idx, -1, next_token)
    return next_token
