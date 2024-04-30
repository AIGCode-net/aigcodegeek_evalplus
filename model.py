import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./hf_home")

import torch
from stop_sequencer import StopSequencer
from vllm import LLM, SamplingParams

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]

class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        direct_completion: bool = True,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        dataset: str = None
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.direct_completion = direct_completion        
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        if direct_completion:
            if dataset.lower() == "humaneval":
                self.eos += ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
            elif dataset.lower() == "mbpp":
                self.eos += ['\n"""', "\nassert"]

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VllmDecoder(DecoderBase):
    def __init__(self, name: str, tp: int, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", tp)),
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs



class AIGCodeGeek(VllmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["direct_completion"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200    
        ) -> List[str]:
        prompt = f"""You are an AI programming assistant, finetuned on the Deepseek Coder base model by AIGCode. You are always willing to answer code-related questions or giving accurate solutions to user instructions.
### Instruction
Write a solution to the following problem:
```python
{prompt}
```
### Response
```python
{prompt}"""
        return VllmDecoder.codegen(self, prompt, do_sample, num_samples)




def make_model(
    model: str,
    template: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    tp=1
):
    assert template in ['AIGCodeGeek'], f"Unsupportede template {template}"
    Bot = eval(template)
    return Bot(
        name=model,
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=1024,
        dataset=dataset,
        tp=tp
    )