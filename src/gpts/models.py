__all__ = [
    'Mistral',
    'Mixtral',
    'OpenOrca',
    'Phi2',
    'OrcaMini3b',
]


import contextlib
import os
import re
import sys
from abc import ABC


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

class Transformers(ABC):
    url_root = 'https://huggingface.co'
    cache_dir = os.path.expanduser('~/.cache/gpts/hf')
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        # self.gpu_is_available = gpu_is_available()

        os.makedirs(self.path(), exist_ok=True)
        # TODO: set context_length

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.url_base, cache_dir=self.cache_dir, **self.tokenizer_args)
        self.model = AutoModelForCausalLM.from_pretrained(self.url_base, cache_dir=self.cache_dir, **self.model_args)


    def url(self):
        return f"{self.url_root}/{self.url_base}"

    def path(self):
        return os.path.expanduser(f'{self.cache_dir}/models--{self.url_base.replace("/", "--")}')
    
    def __call__(self, *args, **kwds):
        return self.ask(*args, **kwds)
    
    def ask(self, question, max_tokens=512):
        raise NotImplementedError

    
class OrcaMini3b(Transformers):
    """ https://huggingface.co/psmathur/orca_mini_3b """
    
    import torch
    url_base = "psmathur/orca_mini_3b"

    tokenizer_args = {}
    model_args = {
        "torch_dtype":torch.float16,
        "device_map":'auto',
    }
    gen_args = {'top_p': 1.0, 'temperature':0.7, 'generate_len': 512, 'top_k': 50}
    system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'

    #generate text function
    def ask(self, instruction, gen_args=None, system=None, input=None):
        if system is None:
            system = self.system
        if gen_args is None:
            gen_args = self.gen_args
        if input:
            prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        else:
            prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

        tokens = self.tokenizer.encode(prompt)
        if gpu_is_available():
            tokens = self.torch.LongTensor(tokens).unsqueeze(0)
            tokens = tokens.to('cuda')

        length = len(tokens[0])
        with self.torch.no_grad():
            rest = self.model.generate(
                input_ids=tokens,
                max_length=length+gen_args['generate_len'],
                use_cache=True,
                do_sample=True,
                top_p=gen_args['top_p'],
                temperature=gen_args['temperature'],
                top_k=gen_args['top_k']
            )
        if not self.verbose:
            output = rest[0][length:]
        string = self.tokenizer.decode(output, skip_special_tokens=True)
        return string


class Phi2(Transformers):
    """ https://huggingface.co/microsoft/phi-2 """
    
    url_base = "microsoft/phi-2"
    tokenizer_args = {
        "torch_dtype":"auto", 
        "trust_remote_code":True
    }
    model_args = {
        "trust_remote_code":True
    }
    gen_args = {}

    def __init__(self, verbose=False):
        if gpu_is_available():
            import torch
            torch.set_default_device("cuda")

        super().__init__(verbose)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def ask(self, question, max_tokens=128, gen_args=None):
        if gen_args is None:
            gen_args = self.gen_args
        inputs = self.tokenizer(question, return_tensors="pt", return_attention_mask=True)
        output = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, max_length=max_tokens, **gen_args)
        output =  self.tokenizer.decode(output[0])
        if not self.verbose:
            output = output[len(question):]
        return output
