__all__ = [
    'Mistral',
    'Mixtral',
    'OpenOrca',
]


import os
import re
import sys
import contextlib


def gpu_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


class Llama:

    def __init__(self, context_length=2048, verbose=False, cpu_only=False):

        self.ensure_have_model()
        self.verbose = verbose
        self.gpu_is_available = gpu_is_available() 
        self.cpu_only = cpu_only
        gpu = gpu_is_available() and not self.cpu_only

        import llama_cpp
        self.llm = llama_cpp.Llama(
            model_path = self.path(),
            n_ctx = context_length,
            n_threads = None if gpu else os.cpu_count(),
            n_gpu_layers = -1 if gpu else 0,
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
            status = os.system(f"wget -O {temp_path!r} {url!r}")
            if status != 0:
                raise RuntimeError(f"Download interrupted. Keeping partial download at {temp_path!r}")

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
            return re.sub(r"[\s\S]*?\[/INST\]\s*", "", output["choices"][0]["text"])


    def ask_for(self, question, type):

        types = (bool, int, float, str, list)
        if type not in types:
            raise TypeError(f"type must be one of: {types!r}")

        import minml
        import guidance
        methods = {
            bool:   minml.gen_bool,
            int:    minml.gen_int,
            float:  minml.gen_float,
            str:    minml.gen_str,
            list:   minml.gen_list,
        }
        method = methods[type]

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            guide = guidance.models.LlamaCpp(self.llm)
            guide += question
            with guidance.block("answer"):
                guide += method()

        return type(guide['answer'])


class Mistral(Llama):

    """ https://mistral.ai/news/announcing-mistral-7b """

    url_root = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main'
    url_base = 'mistral-7b-instruct-v0.2.Q8_0.gguf'


class Mixtral(Llama):

    """ https://mistral.ai/news/mixtral-of-experts """

    url_root = 'https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main'
    url_base = 'mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf'


class OpenOrca(Llama):
    """ https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF"""

    url_root = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main"
    url_base = "mistral-7b-openorca.Q5_K_M.gguf"

### In Progress

class Transformers:
    url_root = 'https://huggingface.co'
    
    def __init__(self, context_length=2048, verbose=False):
        self.context_length = context_length
        self.verbose = verbose
        self.gpu_is_available = gpu_is_available()

        os.makedirs(self.path(), exist_ok=True)
        # TODO: set context_length
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.url_base, cache_dir=self.path())
        self.model = AutoModelForCausalLM.from_pretrained(self.url_base, cache_dir=self.path(), torch_dtype="auto", device_map="auto") 

    def url(self):
        return f"{self.url_root}/{self.url_base}"

    def path(self):
        return os.path.expanduser(f'~/.cache/gpts/hf/models--{self.url_base.replace("/", "--")}')
    
    def __call__(self, *args, **kwds):
        return self.ask(*args, **kwds)
    
    def ask(self, question, max_tokens=512):
        inputs = self.tokenizer(question, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=max_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
class OrcaMini3b(Transformers):
    """ https://huggingface.co/psmathur/orca_mini_3b """
    
    url_base = "psmathur/orca_mini_3b"
