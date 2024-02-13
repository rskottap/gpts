#!/usr/bin/env python

import gpts

def test_mistral_ounces_int():
    question = "How many ounces are in a pound?"
    assert gpts.Mistral().ask_for(question, type=int) == 16

def test_mistral_ounces_float():
    question = "How many ounces are in a pound?"
    assert gpts.Mistral().ask_for(question, type=float) == 16.0

def test_mistral_days_int():
    question = "How many days are in a week?"
    assert gpts.Mistral().ask_for(question, type=int) == 7

def test_mistral_days_float():
    question = "How many days are in a week?"
    assert gpts.Mistral().ask_for(question, type=float) == 7.0

def test_mistral_dozen_int():
    question = "How many eggs are in a dozen?"
    assert gpts.Mistral().ask_for(question, type=int) == 12

def test_mistral_dozen_float():
    question = "How many eggs are in a dozen?"
    assert gpts.Mistral().ask_for(question, type=float) == 12.0

def test_mixtral_ounces_int():
    question = "How many ounces are in a pound?"
    assert gpts.Mixtral().ask_for(question, type=int) == 16

def test_mixtral_ounces_float():
    question = "How many ounces are in a pound?"
    assert gpts.Mixtral().ask_for(question, type=float) == 16.0

def test_mixtral_days_int():
    question = "How many days are in a week?"
    assert gpts.Mixtral().ask_for(question, type=int) == 7

def test_mixtral_days_float():
    question = "How many days are in a week?"
    assert gpts.Mixtral().ask_for(question, type=float) == 7.0

def test_mixtral_dozen_int():
    question = "How many eggs are in a dozen?"
    assert gpts.Mixtral().ask_for(question, type=int) == 12

def test_mixtral_dozen_float():
    question = "How many eggs are in a dozen?"
    assert gpts.Mixtral().ask_for(question, type=float) == 12.0

def test_openorca_ounces_int():
    question = "How many ounces are in a pound?"
    assert gpts.OpenOrca().ask_for(question, type=int) == 16

def test_openorca_ounces_float():
    question = "How many ounces are in a pound?"
    assert gpts.OpenOrca().ask_for(question, type=float) == 16.0

def test_openorca_days_int():
    question = "How many days are in a week?"
    assert gpts.OpenOrca().ask_for(question, type=int) == 7

def test_openorca_days_float():
    question = "How many days are in a week?"
    assert gpts.OpenOrca().ask_for(question, type=float) == 7.0

def test_openorca_dozen_int():
    question = "How many eggs are in a dozen?"
    assert gpts.OpenOrca().ask_for(question, type=int) == 12

def test_openorca_dozen_float():
    question = "How many eggs are in a dozen?"
    assert gpts.OpenOrca().ask_for(question, type=float) == 12.0
