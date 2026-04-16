import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import load_backend_config
from .report_store import build_report_store


class ReportApiHandler(BaseHTTPRequestHandler):
    report_store = None

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "store": getattr(self.report_store, "store_type", "unknown"),
                }
            )
            return

        if parsed.path == "/reports/recent":
            limit = int(query.get("limit", ["20"])[0])
            route = query.get("route", [None])[0]
            zone = query.get("zone", [None])[0]
            reports = self.report_store.list_recent(limit=limit, route=route, zone=zone)
            self._send_json({"count": len(reports), "reports": reports})
            return

        if parsed.path == "/reports/latest":
            route = query.get("route", [None])[0]
            zone = query.get("zone", [None])[0]
            reports = self.report_store.list_recent(limit=1, route=route, zone=zone)
            if not reports:
                self._send_json({"detail": "No reports found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(reports[0])
            return

        if parsed.path.startswith("/reports/"):
            report_id = parsed.path.split("/", 2)[2]
            report = self.report_store.get_report(report_id)
            if report is None:
                self._send_json({"detail": "Report not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(report)
            return

        self._send_json({"detail": "Endpoint not found."}, status=HTTPStatus.NOT_FOUND)


def main():
    config = load_backend_config()
    report_store = build_report_store(config)
    ReportApiHandler.report_store = report_store

    server = ThreadingHTTPServer((config.api_host, config.api_port), ReportApiHandler)
    print(f"API server listening on http://{config.api_host}:{config.api_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
