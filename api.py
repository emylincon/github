from flask import Flask, jsonify
from flask.wrappers import Response
from github import Contributions, Statistics

app: Flask = Flask(__name__)
con_obj: Contributions = Contributions()
VERSION: str = "v1"


@app.route(f"/{VERSION}")
@app.route("/latest")
@app.route("/")
def root() -> Response:
    return jsonify({"api_version": VERSION})


@app.route(f"/{VERSION}/<string:username>/contributions/day/<string:kind>", methods=["GET"])
@app.route("/latest/<string:username>/contributions/day/<string:kind>", methods=["GET"])
def day_contributions(username: str, kind: str) -> Response:
    data: dict = con_obj.get_query(username)
    stat_obj: Statistics = Statistics(data=data)
    response: dict
    if kind == "most":
        response = stat_obj.most_contribution_day().to_dict()
    else:
        response = {"error": f"kind '{kind}' is not supported"}
    return jsonify({"api_version": VERSION, "response": response})
