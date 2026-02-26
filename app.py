"""
app.py — Flask server for SDT Calculator

NO gevent monkey-patching. Uses a plain sync gunicorn worker.

The ShiftML calculation runs in a real OS thread (threading.Thread) so
that PyTorch can import cleanly — gevent's monkey.patch_all() intercepts
os.path.exists at a low level which causes PyTorch to crash on import.

The SSE generator drains a queue from that thread, sending a keepalive
comment every second so proxies and the client don't close the connection
while ShiftML is running (can take 30-120s on first run).
"""

import os, json, traceback, queue, threading
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from sdt_calc import run_calculation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

def allowed_file(f): return "." in f and f.rsplit(".",1)[1].lower() == "cif"

@app.route("/")
def index(): return render_template("index.html")

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/calculate", methods=["POST"])
def calculate():
    if "file" not in request.files: return jsonify({"error":"No file uploaded."}),400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error":"Only .cif files accepted."}),400
    try: cif_text = file.read().decode("utf-8", errors="replace")
    except: return jsonify({"error":"Could not read file."}),400

    nucleus          = request.form.get("nucleus","F").strip().upper()
    disorder         = float(request.form.get("disorder",0.0))
    B0_field         = float(request.form.get("B0_field",9.4))
    N_wanted         = max(100, min(int(request.form.get("N_wanted",1000)), 2000))
    num_orientations = max(1,   min(int(request.form.get("num_orientations",50)), 200))
    mas_rate_khz     = max(0.0, min(float(request.form.get("mas_rate_khz",0.0)), 200.0))
    abund_mode       = request.form.get("abund_mode","natural")
    abund_pct_raw    = request.form.get("abund_pct", None)
    abund_pct        = float(abund_pct_raw) if abund_pct_raw else None

    cs_source = request.form.get("cs_source", "none")
    cs_ics    = float(request.form.get("cs_ics",   0.0))
    cs_delta  = float(request.form.get("cs_delta", 0.0))
    cs_eta    = max(0.0, min(1.0, float(request.form.get("cs_eta", 0.0))))

    # Queue-based threading: run_calculation runs in a real OS thread.
    # The SSE generator below drains results from the queue.
    # None is the sentinel that signals the thread has finished.
    result_queue = queue.Queue()

    def worker():
        try:
            for item in run_calculation(
                cif_text=cif_text, nucleus=nucleus, N_wanted=N_wanted,
                B0_field=B0_field, disorder=disorder,
                num_orientations=num_orientations, mas_rate_khz=mas_rate_khz,
                abund_mode=abund_mode, abund_pct=abund_pct,
                cs_source=cs_source, cs_ics=cs_ics, cs_delta=cs_delta, cs_eta=cs_eta,
            ):
                result_queue.put(("ok", item))
        except Exception as e:
            result_queue.put(("err", str(e)))
        finally:
            result_queue.put(None)  # sentinel

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            try:
                item = result_queue.get(timeout=1.0)
            except queue.Empty:
                # Send a keepalive SSE comment — keeps connection open
                # while ShiftML downloads weights / runs inference
                yield ": keepalive\n\n"
                continue

            if item is None:  # sentinel — thread done
                break

            kind, payload = item
            if kind == "err":
                yield f"data: error:{payload}\n\n"
                break
            if isinstance(payload, str):
                yield f"data: {payload}\n\n"
            else:
                yield f"data: result:{json.dumps(payload)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
