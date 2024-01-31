#!/usr/bin/env python

import os
import sys
from llama_cpp import Llama

class model:

    @classmethod
    def url(cls):
        return f"{cls.url_root}/{cls.url_base}"

    @classmethod
    def path(cls):
        return os.path.expanduser(f'~/.cache/huggingface/{cls.url_base}')

class mistral(model):
    url_root = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main'
    url_base = 'mistral-7b-instruct-v0.2.Q8_0.gguf'

class mixtral(model):
    url_root = 'https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main'
    url_base = 'mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf'



def ensure_have_model(model, path=None):

    cache_path = path or model.path()

    if not os.path.exists(path):
        print(f"Model file {cache_path!r} not found. Downloading.", file=sys.stderr)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        url = model.url()

        # write to a temp file first, so we don't think we have
        # the model already if a partial download is interrupted.
        temp_path = f"{cache_path}.tmp"

        # use wget instead of requests to make buffering
        # easier and give us a progress bar, this file is big.
        os.system(f"wget -O {temp_path!r} {url!r}")

        os.rename(temp_path, cache_path)
        print(f"Model file {cache_path!r} downloaded.", file=sys.stderr)


def ask_llm(question):

    ensure_have_model()

    llm = Llama(
        model_path = cache_path,
        n_ctx = 2048,
        n_threads=os.cpu_count(),
        n_gpu_layers=0,
    )

    output = llm(
        f"[INST] {question} [/INST]",
        max_tokens=512,
        stop=["</s>"],
        echo=True
    )

    return output


def ask_chat_llm(user_prompt, system_prompt=None):

    ensure_have_model()

    llm = Llama(
        model_path = cache_path,
        chat_format="llama-2",
    )

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": "You are a story writing assistant."
        })

    messages.append({
        "role": "user",
        "content": "Write a story about llamas."
    })

    completion = llm.create_chat_completion(messages=messages)
    return completion


if __name__ == '__main__':
    ensure_have_model(mixtral)
    #ask_llm(question)

