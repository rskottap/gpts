import gpts


def ask(model):
    output = model.ask("How many ounces are there in a pound?")
    assert '16' in output


def test_mistral_ask():
    model = gpts.Mistral()
    ask(model)


def test_mixtral_ask():
    model = gpts.Mixtral()
    ask(model)
