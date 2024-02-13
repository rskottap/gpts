#!/usr/bin/env python

import gpts

def ask_mistral_ounces_int():
    question = "How many ounces are in a pound?"
    return (gpts.Mistral().ask_for(question, type=int), 16)

def ask_mistral_ounces_float():
    question = "How many ounces are in a pound?"
    return (gpts.Mistral().ask_for(question, type=float), 16.0)

def ask_mistral_days_int():
    question = "How many days are in a week?"
    return (gpts.Mistral().ask_for(question, type=int), 7)

def ask_mistral_days_float():
    question = "How many days are in a week?"
    return (gpts.Mistral().ask_for(question, type=float), 7.0)

def ask_mistral_dozen_int():
    question = "How many eggs are in a dozen?"
    return (gpts.Mistral().ask_for(question, type=int), 12)

def ask_mistral_dozen_float():
    question = "How many eggs are in a dozen?"
    return (gpts.Mistral().ask_for(question, type=float), 12.0)

def ask_mixtral_ounces_int():
    question = "How many ounces are in a pound?"
    return (gpts.Mixtral().ask_for(question, type=int), 16)

def ask_mixtral_ounces_float():
    question = "How many ounces are in a pound?"
    return (gpts.Mixtral().ask_for(question, type=float), 16.0)

def ask_mixtral_days_int():
    question = "How many days are in a week?"
    return (gpts.Mixtral().ask_for(question, type=int), 7)

def ask_mixtral_days_float():
    question = "How many days are in a week?"
    return (gpts.Mixtral().ask_for(question, type=float), 7.0)

def ask_mixtral_dozen_int():
    question = "How many eggs are in a dozen?"
    return (gpts.Mixtral().ask_for(question, type=int), 12)

def ask_mixtral_dozen_float():
    question = "How many eggs are in a dozen?"
    return (gpts.Mixtral().ask_for(question, type=float), 12.0)

def ask_openorca_ounces_int():
    question = "How many ounces are in a pound?"
    return (gpts.OpenOrca().ask_for(question, type=int), 16)

def ask_openorca_ounces_float():
    question = "How many ounces are in a pound?"
    return (gpts.OpenOrca().ask_for(question, type=float), 16.0)

def ask_openorca_days_int():
    question = "How many days are in a week?"
    return (gpts.OpenOrca().ask_for(question, type=int), 7)

def ask_openorca_days_float():
    question = "How many days are in a week?"
    return (gpts.OpenOrca().ask_for(question, type=float), 7.0)

def ask_openorca_dozen_int():
    question = "How many eggs are in a dozen?"
    return (gpts.OpenOrca().ask_for(question, type=int), 12)

def ask_openorca_dozen_float():
    question = "How many eggs are in a dozen?"
    return (gpts.OpenOrca().ask_for(question, type=float), 12.0)

if __name__ == '__main__':
    funcs = [v for k,v in globals().items() if k.startswith('ask')]
    for func in funcs:
        print(f"{func.__name__}: {func()}")
