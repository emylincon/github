import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Prep:
    def prepx(self, x: list, degree: int, include_bias: bool = False):
        if len(np.array(x).shape) == 1:
            x = np.array(x).reshape((-1, 1))
        transformer = PolynomialFeatures(
            degree=degree, include_bias=include_bias)
        return transformer.fit_transform(x)


class Predictor(Prep):
    def __init__(self, model, degree: int) -> None:
        self.model = model
        self.degree = degree

    def predict(self, x: list):
        x_ = self.prepx(x, self.degree)
        return self.model.predict(x_)


class Model:
    def __init__(self, x, y, degree, model):
        self.x = x
        self.y = y
        self.degree = degree
        self.model = model
        self.score = self.model.score(self.x, y)


class BestModel(Prep):
    def __init__(self, x: list, y: list, max_compare_length: int = 10) -> None:
        self.x = x
        self.y = np.array(y)
        self.max_compare_length = max_compare_length

    def get_model(self, degree: int):
        x_ = self.prepx(self.x, degree)
        model = LinearRegression().fit(x_, self.y)
        return Model(x_, self.y, degree, model)

    def compute_best_model(self):
        models = [self.get_model(i) for i in range(2, self.max_compare_length)]
        # model_scores = {m.degree: m.score for m in models}
        # print(model_scores)
        best = None
        for m in models:
            if best is None:
                best = m
                continue
            if m.score == best.score:
                continue
            elif m.score == 1:
                break
            elif m.score >= 0.95 and best.score >= 0.95:
                tmp = {m: m.score, best: best.score}
                best = min(tmp, key=tmp.get)
            else:
                tmp = {m: m.score, best: best.score}
                best = max(tmp, key=tmp.get)

        # print("best score =", best.score, "degree =", best.degree)
        return Predictor(best.model, best.degree)


if __name__ == "__main__":
    xi = [5, 15, 25, 35, 45, 55]
    yi = [5, 20, 14, 32, 22, 38]
    my_obj = BestModel(xi, yi)
    p = my_obj.compute_best_model()
    p.predict([25])
