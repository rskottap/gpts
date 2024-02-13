import gpts

QUESTIONS_AND_ANSWERS = {
    "How many ounces are there in a pound?": '16',
    "How many eggs are in a dozen?": '12',
    "How many days are in a week?": '7',
}

def ask_typed_questions(model):
    for question, answer in QUESTIONS_AND_ANSWERS.items():
        assert abs(model.ask_for(question, type=int))   == int(answer)
        assert abs(model.ask_for(question, type=float)) == float(answer)

def test_mistral():
    ask_typed_questions(gpts.Mistral())

def test_mixtral():
    ask_typed_questions(gpts.Mixtral())

def test_openorca():
    ask_typed_questions(gpts.OpenOrca())
