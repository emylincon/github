import json
import requests
from jinja2 import Template
import os
import pandas as pd
from datetime import datetime


class Contributions:
    def __init__(self):
        self.url = 'https://api.github.com/graphql'
        self.token = os.getenv('GITHUB_PERSONAL_TOKEN')
        self.header = {'Authorization': f'bearer {self.token}'}

    @staticmethod
    def get_date_range(start, end):
        if start:
            start, end = start.strftime('%Y-%m-%dT%H:%M:%SZ'), end.strftime('%Y-%m-%dT%H:%M:%SZ')
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
        query = {"query": f"{self.get_query_data(username, start_date, end_date)}"}
        response = requests.post(self.url, json=query, headers=self.header)
        with open("res.json", "w") as f:
            json.dump(response.json(), f)
        return response.json()


class Transform:
    def __init__(self, data):
        self.data = data
        self.tf_data = None

    def days_contribution(self):
        data = []
        for week in self.data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                data.append({"date": day["date"], "contribution": day["contributionCount"]})
        df = pd.DataFrame(data)
        df.date = pd.to_datetime(df.date, infer_datetime_format=True)
        df['day_name'] = [df.iloc[i].date.day_name() for i in range(len(df))]
        self.tf_data = df

    def most_contribution_day(self):
        id_max = self.tf_data.contribution.idxmax()
        return self.tf_data.iloc[id_max]

    def weekday_contributions(self):
        return self.tf_data.groupby(by="day_name").sum().sort_values(by=['contribution'], ascending=False)

    def most_weekday_contributions(self):
        df = self.weekday_contributions()
        return df.iloc[0]


if __name__ == "__main__":
    obj = Contributions()
    dd = obj.get_query("emylincon")
    tr = Transform(dd)
    tr.days_contribution()
