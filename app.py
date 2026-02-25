"""
app.py — Flask web server for the SDT Calculator
SSE streaming endpoint for live orientation progress.
"""
import os, json, traceback
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

    def generate():
        try:
            for item in run_calculation(
                cif_text=cif_text, nucleus=nucleus, N_wanted=N_wanted,
                B0_field=B0_field, disorder=disorder,
                num_orientations=num_orientations, mas_rate_khz=mas_rate_khz,
                abund_mode=abund_mode, abund_pct=abund_pct,
            ):
                if isinstance(item, str):
                    yield f"data: {item}\n\n"
                else:
                    yield f"data: result:{json.dumps(item)}\n\n"
        except ValueError as e:
            yield f"data: error:{e}\n\n"
        except Exception:
            traceback.print_exc()
            yield "data: error:Calculation failed. Check your CIF and parameters.\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
