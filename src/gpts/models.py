__all__ = [
    'Mistral',
    'Mixtral',
]


import os
import re
import sys

from llama_cpp import Llama


def gpu_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


class Model:

    """
        Leave the public interface here underspecified until we have
        at least one non-mistrell.ai model we want to support.
        We'll learn the right abstractions as we go.
    """

    def __init__(self, context_length=2048, verbose=False):

        self.ensure_have_model()
        self.verbose = verbose
        self.gpu_is_available = gpu_is_available() 

        self.llm = Llama(
            model_path = self.path(),
            n_ctx = context_length,
            n_threads = None if self.gpu_is_available else os.cpu_count(),
            n_gpu_layers = -1 if self.gpu_is_available else 0,
            verbose = self.verbose,
        )

    def url(self):
        return f"{self.url_root}/{self.url_base}"

    def dirname(self):
        name = self.__class__.__name__.lower()
        return os.path.expanduser(f'~/.cache/gpts/{name}')

    def path(self):
        return os.path.join(self.dirname(), self.url_base)

    def ensure_have_model(self, path=None):

        cache_path = path or self.path()

        if not os.path.exists(cache_path):
            print(f"Model file {cache_path!r} not found. Downloading.", file=sys.stderr)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            url = self.url()

            # write to a temp file first, so we don't think we have
            # the model already if a partial download is interrupted.
            temp_path = f"{cache_path}.tmp"

            # use wget instead of requests to make buffering
            # easier and give us a progress bar, this file is big.
            os.system(f"wget -O {temp_path!r} {url!r}")

            os.rename(temp_path, cache_path)
            print(f"Model file {cache_path!r} downloaded.", file=sys.stderr)


    def __call__(self, *args, **kwds):
        return self.ask(*args, **kwds)

    def ask(self, question, max_tokens=512):

        output = self.llm(
            f"[INST] {question} [/INST]",
            max_tokens=max_tokens,
            stop=["</s>"],
            echo=True,
        )
        if self.verbose:
            return output
        else:
            return re.sub(r"[\s\S]*?\[/INST\] ", "", output["choices"][0]["text"])

    def chat(self, user_prompt, system_prompt=None):

        # TODO: Do we need to create a new instance here? It's extremely quick
        # so this may be simpler than keeping the state. Can we do this for .ask()?

        llm = Llama(
            model_path = self.path(),
            chat_format="llama-2",
            verbose = self.verbose,
        )

        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        output = llm.create_chat_completion(messages=messages)

        if self.verbose:
            return output
        else:
            return output['choices'][0]['message']['content'].strip()


class Mistral(Model):

    """ https://mistral.ai/news/announcing-mistral-7b """

    url_root = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main'
    url_base = 'mistral-7b-instruct-v0.2.Q8_0.gguf'


class Mixtral(Model):

    """ https://mistral.ai/news/mixtral-of-experts """

    url_root = 'https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main'
    url_base = 'mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf'

from transformers import AutoModelForCausalLM, AutoTokenizer


class Tranformers():
    url_root = 'https://huggingface.co'
    
    def __init__(self, context_length=2048, verbose=False):
        self.context_length = context_length
        self.verbose = verbose
        self.gpu_is_available = gpu_is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(self.url_base)
        self.model = AutoModelForCausalLM.from_pretrained(self.url_base, torch_dtype="auto", context_length=self.context_length, verbose=self.verbose)

    def url(self):
        return f"{self.url_root}/{self.url_base}"

    def path(self):
        return os.path.expanduser(f'~/.cache/huggingface/hub/models--{self.url_base.replace("/", "--")}')
    
    def __call__(self, *args, **kwds):
        return self.ask(*args, **kwds)
    
    def ask(self, question, max_tokens=512):
        inputs = self.tokenizer(question, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=max_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
class Phi2(Tranformers):
    """ https://huggingface.co/microsoft/phi-2 """
    
    url_base = "microsoft/phi-2"