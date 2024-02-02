import gpts


def query_text(model):
    q = "Extract the name, age, role and company of the person. Return these as a python dictionary with no other text or explanation."
    output = model.query(q, "./files/example.txt")
    assert "Alex" in output
    assert "Amazon" in output

def query_pdf(model):
    q = "What is it called when kneecaps slide out of place?"
    output = model.query(q, "./files/pomeranian.pdf")
    assert "patellar luxation" in output.lower()

def query_image(model):
    i = " Return only the answer to the question. Do not return any other text or explanation."
    q1 = "How many cats?"
    q2 = "What sport is being played?"

    output1 = model.query(q1 + i, "./files/cats.jpg").lower()
    output2 = model.query(q2 + i, "./files/soccer.jpg").lower()
    assert ('2' in output1) | ('two' in output1)
    assert 'soccer' in output2


def test_mixtral_query():
    model = gpts.Mixtral()
    query_text(model)
    query_pdf(model)
    query_image(model)

def test_mistral_query():
    model = gpts.Mistral()
    query_text(model)
    query_pdf(model)
    query_image(model)
