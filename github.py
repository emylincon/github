import json
import requests
from jinja2 import Template
import os
import pandas as pd
import datetime
from Regression import BestModel


class Contributions:
    def __init__(self):
        self.url = 'https://api.github.com/graphql'
        self.token = os.getenv('GITHUB_PERSONAL_TOKEN')
        self.header = {'Authorization': f'bearer {self.token}'}

    @staticmethod
    def get_date_range(start, end):
        if start:
            start, end = start.strftime(
                '%Y-%m-%dT%H:%M:%SZ'), end.strftime('%Y-%m-%dT%H:%M:%SZ')
            return f'(from: "{start}", to: "{end}")'
        else:
            return ""

    @property
    def template(self):
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

    def get_query_data(self, username, start, end):
        t = Template(self.template)
        return t.render(username=username, date_range=self.get_date_range(start, end))

    def get_query(self, username, start_date=None, end_date=None):
        query = {
            "query": f"{self.get_query_data(username, start_date, end_date)}"}
        response = requests.post(self.url, json=query, headers=self.header)
        return response.json()


class Statistics:
    def __init__(self, data):
        self.data = data
        self.total_contributions = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["totalContributions"]
        self.tf_data = self.days_contribution()

    def days_contribution(self):
        data = []
        for week in self.data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                data.append(
                    {"date": day["date"], "contribution": day["contributionCount"]})
        df = pd.DataFrame(data)
        df.date = pd.to_datetime(df.date, infer_datetime_format=True)
        df['day'] = [df.iloc[i].date.day_name() for i in range(len(df))]
        df['month'] = [df.iloc[i].date.month_name() for i in range(len(df))]
        return df

    def most_contribution_day(self):
        id_max = self.tf_data.contribution.idxmax()
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
    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data
        self.model = None
        self.last = 1
        self.get_model()

    def data_prep(self):
        x_series = []
        y_series = []
        diff = 0
        for week in self.raw_data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                date = datetime.datetime.strptime(day["date"], '%Y-%m-%d')
                x_series.append([date.month, date.day, self.last])
                if day["contributionCount"] == 0:
                    diff += 1
                    y_series.append(diff)
                else:
                    diff = 0
                    y_series.append(diff)
                self.last += 1
        return x_series, y_series

    def get_model(self):
        x, y = self.data_prep()
        self.model = BestModel(x, y).compute_best_model()

    def predict_next(self):
        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
        input_data = [[tomorrow.month, tomorrow.day, self.last]]
        raw_result = self.model.predict(input_data)
        result = abs(round(raw_result[0]))
        date = tomorrow + datetime.timedelta(days=result)
        return {"days": result, "date": date}


if __name__ == "__main__":
    obj = Contributions()
    dd = obj.get_query("emylincon")
    tr = Statistics(dd)
    print(tr.most_contribution_day())
    ml = ML(dd)
    print("ML =>", ml.predict_next())
