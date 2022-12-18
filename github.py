import json
import requests
from jinja2 import Template
import os
import pandas as pd
import datetime as dt
from Regression import BestModel
from math import ceil
from typing import Any

default_date: dt.datetime = dt.datetime(1, 1, 1, 0, 0)


class Contributions:
    def __init__(self):
        self.url: str = 'https://api.github.com/graphql'
        self.token: str = os.getenv('GITHUB_PERSONAL_TOKEN')
        self.header: dict = {'Authorization': f'bearer {self.token}'}

    @staticmethod
    def get_date_range(start: dt.datetime, end: dt.datetime) -> str:
        if start == default_date:
            def strf(x): return x.strftime('%Y-%m-%dT%H:%M:%SZ')
            return f'(from: "{strf(start)}", to: "{strf(end)}")'
        else:
            return ""

    @property
    def template(self) -> str:
        return """query {
        user(login: "{{ username }}") {
          email
          createdAt
          contributionsCollection {{ date_range }} {
            contributionCalendar {
              totalContributions
              weeks {
                contributionDays {
                  weekday
                  date
                  contributionCount
                  color
                }
              }
              months  {
                name
                  year
                  firstDay
                totalWeeks

              }
            }
          }
        }

      }
      """

    def get_query_data(self, username: str, start: dt.datetime, end: dt.datetime) -> str:
        t: Template = Template(self.template)
        return t.render(username=username, date_range=self.get_date_range(start, end))

    def get_query(self, username: str, start_date: dt.datetime = default_date, end_date: dt.datetime = default_date) -> dict:
        query: dict = {
            "query": f"{self.get_query_data(username, start_date, end_date)}"}
        response: requests.Response = requests.post(
            self.url, json=query, headers=self.header)
        if response.status_code != 200:
            return {"error": json.loads(response.content.decode("utf-8"))}
        return response.json()


class Statistics:
    def __init__(self, data: dict):
        self.data: dict = data
        self.total_contributions: int = data["data"]["user"][
            "contributionsCollection"]["contributionCalendar"]["totalContributions"]
        self.tf_data: pd.DataFrame = self.days_contribution()

    def days_contribution(self) -> pd.DataFrame:
        data: list = []
        for week in self.data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                data.append(
                    {"date": day["date"], "contribution": day["contributionCount"]})
        df: pd.DataFrame = pd.DataFrame(data)
        df.date = pd.to_datetime(df.date, infer_datetime_format=True)
        df['day'] = [df.iloc[i].date.day_name() for i in range(len(df))]
        df['month'] = [df.iloc[i].date.month_name() for i in range(len(df))]
        return df

    def most_contribution_day(self) -> pd.DataFrame:
        id_max: int = self.tf_data.contribution.idxmax()
        return self.tf_data.iloc[id_max]

    def weekday_contributions(self):
        return self.tf_data.groupby(by="day").sum().sort_values(by=['contribution'], ascending=False)

    def most_weekday_contributions(self):
        df = self.weekday_contributions()
        return df.iloc[0]

    def month_contributions(self):
        return self.tf_data.groupby(by="month").sum().sort_values(by=['contribution'], ascending=False)

    def most_month_contributions(self):
        df = self.month_contributions()
        return df.iloc[0]

    def least_month_contributions(self):
        df = self.month_contributions()
        return df.iloc[-1]

    def average_contribution_per_month(self) -> int:
        return round(self.total_contributions/12)

    def average_contribution_per_week(self) -> int:
        return round(self.total_contributions/52)


class ML:
    def __init__(self, raw_data, max_compare_length: int = 20) -> None:
        self.raw_data = raw_data
        self.model = None
        self.max_compare_length = max_compare_length
        self.get_model()

    def data_prep(self):
        return None

    def get_model(self):
        prepared = self.data_prep()
        if prepared:
            x, y = prepared
            self.model = BestModel(
                x, y, self.max_compare_length).compute_best_model()
            return
        print("[warning]: self.data_prep() returned None")


class PredictNext(ML):
    """
     Predict when the date of next contribution: use distance between contributions as training data
    """

    def __init__(self, raw_data) -> None:
        self.last = 1
        self.max_result_days = 365
        super().__init__(raw_data=raw_data)

    def data_prep(self):
        x_series = []
        y_series = []
        diff = 0
        for week in self.raw_data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                date = dt.datetime.strptime(day["date"], '%Y-%m-%d')
                x_series.append([date.month, date.day, self.last])
                if day["contributionCount"] == 0:
                    diff += 1
                    y_series.append(diff)
                else:
                    diff = 0
                    y_series.append(diff)
                self.last += 1
        return x_series, y_series

    def predict_next(self) -> dict:
        if self.model:
            tomorrow = dt.datetime.now() + dt.timedelta(days=1)
            input_data = [[tomorrow.month, tomorrow.day, self.last]]
            raw_result = self.model.predict(input_data)
            result = abs(round(raw_result[0]))
            if result > self.max_result_days:
                result = self.max_result_days
            date = tomorrow + dt.timedelta(days=result)
            return {"days": result, "date": date}
        return {"error": "model has not been trained"}


class PredictTotalWeek(ML):
    """
    Predict total contributions for a given week
    """

    def __init__(self, raw_data) -> None:
        super().__init__(raw_data=raw_data)

    def week_of_month(self, dt):
        """
        Returns the week of the month for the specified date.
        """
        first_day = dt.replace(day=1)

        dom = dt.day
        adjusted_dom = dom + first_day.weekday()

        return int(ceil(adjusted_dom/7.0))

    def data_prep(self):
        x_series = []
        y_series = []
        current_week_date = None
        current_week_contribution = 0

        for week in self.raw_data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                date = dt.datetime.strptime(day["date"], '%Y-%m-%d')
                if current_week_date is None:
                    current_week_date = date
                # If dates are in same week
                if date.isocalendar()[1] == current_week_date.isocalendar()[1]:
                    current_week_contribution += day["contributionCount"]
                else:
                    if current_week_contribution == 0:
                        current_week_date = date
                        continue
                    row = [self.week_of_month(current_week_date), current_week_date.isocalendar()[
                        1], current_week_date.month]
                    x_series.append(row)
                    y_series.append(current_week_contribution)
                    # reset current week contribution
                    current_week_contribution = day["contributionCount"]
                    current_week_date = date
        self.max_week_contribution = max(y_series)
        return x_series, y_series

    def predict_week(self, week_date: dt.datetime) -> dict:
        if self.model:
            row = [self.week_of_month(week_date), week_date.isocalendar()[
                1], week_date.month]
            input_data = [row]
            raw_result = self.model.predict(input_data)
            result = abs(round(raw_result[0]))
            if result > (2*self.max_week_contribution):
                result = 0
            return {"totalPredictedContribution": result, "weekdate": week_date}
        return {"error": "model is None"}


class PredictTotalMonth(ML):
    """
    Predict total contributions for a given month
    """

    def __init__(self, raw_data) -> None:
        super().__init__(raw_data=raw_data)

    def data_prep(self):
        x_series = []
        y_series = []
        current_month = None
        current_month_contribution = 0

        for week in self.raw_data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                date = dt.datetime.strptime(day["date"], '%Y-%m-%d')
                if current_month is None:
                    current_month = date.month
                # If dates are in same month
                if date.month == current_month:
                    current_month_contribution += day["contributionCount"]
                else:
                    if current_month_contribution == 0:
                        current_month = date.month
                        continue
                    row = [current_month]
                    x_series.append(row)
                    y_series.append(current_month_contribution)
                    # reset current month contribution
                    current_month_contribution = day["contributionCount"]
                    current_month = date.month
        self.max_month_contribution = max(y_series)
        return x_series, y_series

    def predict_month(self, month: int) -> dict:
        if self.model:
            input_data: list[int] = [month]
            raw_result = self.model.predict(input_data)
            result = abs(round(raw_result[0]))
            if result > (3*self.max_month_contribution):
                result = 0
            return {"totalPredictedContribution": result, "month": month}
        return {"error": "model is None"}


class PredictTotalYear:
    """
    Predict total contributions for the current year
    """

    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data
        self.monthModel = PredictTotalMonth(raw_data=raw_data)

    def predict(self) -> dict:
        def get_value(no): return abs(
            round(self.monthModel.predict_month(no)["totalPredictedContribution"]))
        result = sum(get_value(i) for i in range(1, 13))
        return {"totalPredictedContribution": result, "year": dt.datetime.now().year}


if __name__ == "__main__":
    obj = Contributions()
    dd = obj.get_query("emylincon")
    # tr = Statistics(dd)
    # print(tr.most_contribution_day())
    # ml = PredictNext(dd)
    # print("ML =>", ml.predict_next())
    # mlw = PredictTotalWeek(dd)
    # for i in range(1, 49, 7):
    #     next_week = dt.datetime.now() + dt.timedelta(days=i)
    #     print("MLW =>", mlw.predict_week(next_week))
    mlm = PredictTotalMonth(dd)
    print(mlm.predict_month(4))
    mly = PredictTotalYear(dd)
    print(mly.predict())
