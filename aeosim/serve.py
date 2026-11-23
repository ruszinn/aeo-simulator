"""Browse one scenario's mock web in a real browser (for debugging and demos).

Pages live at http://localhost:<port>/<domain>/<path>. The index page lists
every firm with its feature switches and links to what a crawler, a plain
fetcher and a JavaScript-running browser would each see.
"""

from __future__ import annotations

import html

from flask import Flask, Response, abort


def serve(web, scenario, port: int = 8000) -> None:
    app = Flask(__name__)

    @app.route("/")
    def index():
        rows = []
        for m in scenario.market_record():
            on = ", ".join(k for k, v in m["features"].items() if v) or "none"
            d = m["domain"]
            rows.append(f"<tr><td>{html.escape(m['name'])}</td><td>{m['fit']['full_fit']}</td><td>{on}</td>"
                        f"<td><a href='/{d}/'>rendered</a> | <a href='/raw/{d}/'>initial HTML</a> | "
                        f"<a href='/agent/{d}/'>agent view</a></td></tr>")
        return (f"<h2>Scenario {scenario.scenario_id}</h2><p>{html.escape(scenario.brief.text)}</p>"
                "<table border=1 cellpadding=4><tr><th>Firm</th><th>Full fit</th><th>Features on</th><th>Views</th></tr>"
                + "".join(rows) + "</table>")

    def _res(domain, path):
        r = web.get(f"https://{domain}/{path}")
        if r is None:
            abort(404)
        return r

    @app.route("/<domain>/", defaults={"path": ""})
    @app.route("/<domain>/<path:path>")
    def rendered(domain, path):
        r = _res(domain, path)
        return Response(r.rendered, mimetype="text/html" if r.content_type == "html" else "text/plain")

    @app.route("/raw/<domain>/", defaults={"path": ""})
    @app.route("/raw/<domain>/<path:path>")
    def raw(domain, path):
        return Response(_res(domain, path).initial, mimetype="text/plain")

    @app.route("/agent/<domain>/", defaults={"path": ""})
    @app.route("/agent/<domain>/<path:path>")
    def agent(domain, path):
        return Response(web.fetch(f"https://{domain}/{path}")["text"], mimetype="text/plain")

    print(f"Serving scenario {scenario.scenario_id} at http://localhost:{port}/")
    app.run(port=port)
