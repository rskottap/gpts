import gpts

def ask_for_truth(model):
    output = model.ask("How many ounces are there in a pound?")
    assert '16' in output

def ask_for_lies(model):
    output = model.chat(
        user_prompt = "How many ounces are there in a pound?",
        system_prompt = "I want you to subtract 1 from each answer before returning it."
    )
    assert '15' in output

def test_mistral_ask():
    model = gpts.Mistral()
    ask_for_truth(model)

def test_mistral_chat():
    model = gpts.Mistral()
    ask_for_lies(model)

def test_mixtral_ask():
    model = gpts.Mixtral()
    ask_for_truth(model)

def test_mixtral_chat():
    model = gpts.Mixtral()
    ask_for_lies(model)
