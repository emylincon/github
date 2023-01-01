from flask import Flask, jsonify
from flask.wrappers import Response
from github import Contributions, Statistics
import json
from dotenv import load_dotenv
import os

app: Flask = Flask(__name__)
con_obj: Contributions = Contributions()
VERSION: str = "v1"
load_dotenv('.env')
ENVIRONMENT: str = "Development" if (
    n := os.getenv('ENVIRONMENT')) is None else str(n)


def get_data(username: str) -> dict:
    if ENVIRONMENT.lower() == "test":
        with open("test/data.json") as file_tmp:
            return json.load(file_tmp)
    else:
        return con_obj.get_query(username)


@app.route(f"/{VERSION}")
@app.route("/latest")
@app.route("/")
def root() -> Response:
    return jsonify({"api_version": VERSION})


@app.route(f"/{VERSION}/<string:username>/contributions/day/<string:kind>", methods=["GET"])
@app.route("/latest/<string:username>/contributions/day/<string:kind>", methods=["GET"])
def day_contributions(username: str, kind: str) -> Response:
    data: dict = get_data(username)

    stat_obj: Statistics = Statistics(data=data)
    response: dict
    if kind == "most":
        response = stat_obj.most_contribution_day().to_dict()
    elif kind == "average":
        response = {"average_day_contribution": stat_obj.avg_contribution_day()}
    elif kind == "least":
        response = stat_obj.least_contribution_day().to_dict()
    else:
        response = {"error": f"kind '{kind}' is not supported"}
    return jsonify({"api_version": VERSION, "response": response})
