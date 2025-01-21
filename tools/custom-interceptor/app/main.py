from flask import Flask, request, jsonify
import json
import requests
import os

app = Flask(__name__)

@app.route("/", methods=["POST"])
def forward_request():

    raw_body = request.data
    payload = json.loads(raw_body)

    x_github_event = request.headers.get("X-GitHub-Event", "not provided")
    jwt_token = request.headers.get("JWT-Token", "not provided")

    ## VALIDATE THE TOKEN
    auth_url = os.getenv("AUTH_URL")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    try:
        response = requests.post(
            auth_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "token": jwt_token,
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({
            "status": 500,
            "message": f"Encounted an error: {e}"
        })

    response_data = response.json()
    if response_data.get("active") is True:
        try:
            response = requests.post(
                os.getenv("WEBHOOK_URL"),
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": f"{x_github_event}",
                },
                data=payload,
            )
            return jsonify({
                "status": 200,
                "message": "invalid event body format: unexpected end of JSON input"
            })
        except:
            return jsonify({
                "status": 500,
                "message": f"Unable to call PaC."
            })
    else:
        return jsonify({
            "status": 401,
            "message": "JWT token is invalid or expired."
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
