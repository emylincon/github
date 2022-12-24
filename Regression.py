import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Any
from functools import cache


class Prep:
    @cache
    def prepx(self, x: tuple, degree: int, include_bias: bool = False):
        if len(np.array(x).shape) == 1:
            x = np.array(x).reshape((-1, 1)).tolist()
        transformer = PolynomialFeatures(
            degree=degree, include_bias=include_bias)
        return transformer.fit_transform(x)


class Predictor(Prep):
    def __init__(self, model: LinearRegression, degree: int) -> None:
        self.model: LinearRegression = model
        self.degree: int = degree

    @cache
    def predict(self, x: tuple) -> np.ndarray:
        x_ = self.prepx(x, self.degree)
        return self.model.predict(x_)


class Model:
    def __init__(self, x: Any, y: np.ndarray, degree: int, model: LinearRegression):
        self.x: Any = x
        self.y: np.ndarray = y
        self.degree: int = degree
        self.model: LinearRegression = model
        self.score: float = float(self.model.score(self.x, y))


class BestModel(Prep):
    def __init__(self, x: tuple, y: tuple, max_compare_length: int = 10) -> None:
        self.x: Any = x
        self.y: np.ndarray = np.array(y)
        self.max_compare_length = max_compare_length

    def get_model(self, degree: int) -> Model:
        x_ = self.prepx(self.x, degree)
        model: LinearRegression = LinearRegression().fit(x_, self.y)
        return Model(x_, self.y, degree, model)

    def compute_best_model(self) -> Predictor:
        models = [self.get_model(i) for i in range(2, self.max_compare_length)]

        best: Model = models[0]
        for m in models[1:]:
            if m.score == best.score:
                continue
            elif m.score == 1:
                break
            elif m.score >= 0.95 and best.score >= 0.95:
                tmp: dict[Model, float] = {m: m.score, best: best.score}
                best = min(tmp, key=lambda x: tmp[x])
            else:
                tmp = {m: m.score, best: best.score}
                best = max(tmp, key=lambda x: tmp[x])
        return Predictor(best.model, best.degree)


NONE_PREDICTOR: Predictor = Predictor(
    LinearRegression().fit(Prep().prepx((1,), 1), np.array([1])), 0)

if __name__ == "__main__":
    xi = (5, 15, 25, 35, 45, 55)
    yi = (5, 20, 14, 32, 22, 38)
    my_obj = BestModel(xi, yi)
    p = my_obj.compute_best_model()
    p.predict((25,))
