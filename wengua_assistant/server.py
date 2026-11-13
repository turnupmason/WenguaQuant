"""Loopback-only browser interface. No hosted endpoint or training API."""
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import hmac, json, secrets, threading, webbrowser
from .service import Service

STATIC = Path(__file__).with_name('static')


def handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): pass

        def reply(self, status, value, mime='application/json; charset=utf-8'):
            raw = json.dumps(value, ensure_ascii=False).encode('utf8') if mime.startswith('application/json') else value
            self.send_response(status); self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            try:
                self.end_headers(); self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                # Navigating away closes an HTTP client, not the model job.
                pass

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.valid_host(): return self.reply(403, {'error': '仅允许本机访问'})
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/':
                    return self.reply(200, (STATIC / 'index.html').read_bytes().replace(b'__TOKEN__', token.encode()), 'text/html; charset=utf-8')
                if parsed.path in {'/app.js', '/style.css'}:
                    mime = 'text/javascript' if parsed.path.endswith('.js') else 'text/css'
                    return self.reply(200, (STATIC / parsed.path[1:]).read_bytes(), mime + '; charset=utf-8')
                # API calls require a token: other local web pages cannot read
                # private book passages through ordinary cross-site requests.
                if not hmac.compare_digest(self.headers.get('X-Wengua-Token', ''), token):
                    return self.reply(403, {'error': '请从启动页访问'})
                if parsed.path == '/api/info':
                    from .sources import case_list
                    from wengua_course.data import DOMAIN_ZH
                    from .inference import ADAPTER
                    candidate = ADAPTER.is_dir()
                    return self.reply(200, {'domains': DOMAIN_ZH, 'cases': case_list(), 'candidate_present': candidate,
                                            'training_updates': 0, 'professional_approved': False})
                if parsed.path == '/api/job':
                    return self.reply(200, service.poll(parse_qs(parsed.query).get('id', [''])[0]))
                if parsed.path == '/api/export':
                    return self.reply(200, service.export(parse_qs(parsed.query).get('session', [''])[0]), 'application/zip')
                if parsed.path == '/api/case':
                    from .sources import case_source
                    return self.reply(200, case_source(parse_qs(parsed.query).get('id', [''])[0]))
                return self.reply(404, {'error': '页面不存在'})
            except Exception as error: return self.reply(400, {'error': str(error)})

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.valid_host() or origin not in allowed or not hmac.compare_digest(self.headers.get('X-Wengua-Token', ''), token):
                return self.reply(403, {'error': '请求来源不正确'})
            try:
                if self.path != '/api/ask': return self.reply(404, {'error': '接口不存在'})
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32000: raise ValueError('问题资料过长')
                if self.headers.get_content_type() != 'application/json': raise ValueError('只接收JSON')
                payload = json.loads(self.rfile.read(size).decode('utf8'))
                return self.reply(202, service.submit(payload))
            except Exception as error: return self.reply(400, {'error': str(error)})
    return Handler


def main():
    parser = ArgumentParser(description='现有8B模型＋程序排盘＋两书来源；只做推理，不训练')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    from wengua_course.runtime import configure
    configure()
    from .sources import checked_index
    checked_index()
    service = Service(); token = secrets.token_hex(32)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(service, token))
    url = f'http://127.0.0.1:{server.server_port}'
    print('Wengua 本地问卦：' + url, flush=True)
    print('使用现有权重。没有训练任务。每答独立进程，原始答复与依据保存在 local/assistant/jobs。', flush=True)
    if not args.no_browser: threading.Timer(.8, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print('\n正在结束本地服务。', flush=True)
    finally: server.server_close(); service.close()
    return 0


if __name__ == '__main__': raise SystemExit(main())
