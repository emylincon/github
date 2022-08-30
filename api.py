import json
from urllib import response
from flask import Flask, jsonify
from github import Contributions, Statistics

app = Flask(__name__)
con_obj = Contributions()

VERSION = "v1"


@app.route("/")
def root():
    return jsonify({"api_version": VERSION})


@app.route(f"/{VERSION}/<string:username>/contributions/day/<string:kind>", methods=["GET"])
def day_contributions(username, kind):
    data = con_obj.get_query(username)
    with open("emeka", "w") as file:
        json.dump(data, file, indent=4)

    stat_obj = Statistics(data=data)
    response = ""
    if kind == "most":
        response = stat_obj.most_contribution_day().to_dict()
    else:
        response = f"kind '{kind}' is not supported"
    return jsonify({"api_version": VERSION, "response": response})
