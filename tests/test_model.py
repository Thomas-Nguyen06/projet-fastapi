from api.app import load_model

model = load_model()


def test_coherency_results():
    x = [["100", "3", "1"]]
    assert(model.predict(x)[0] > 10000)

    x = [["100", "3", "-10"]]
    assert(model.predict(x)[0] < 0)

    x = [["10000", "20", "1"]]
    assert(model.predict(x)[0] > 1000000)
