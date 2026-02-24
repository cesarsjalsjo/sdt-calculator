"""
app.py — Flask web server for the SDT Calculator

Endpoints
---------
GET  /          → Serves the upload HTML page
POST /calculate → Accepts a CIF file upload, runs the calculation, returns JSON
GET  /health    → Simple health check for deployment platforms
"""

import os
import json
import traceback
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from sdt_calc import run_calculation

# =============================================================================
#  APP CONFIGURATION
# =============================================================================

app = Flask(__name__)

# Maximum upload size: 5 MB (CIF files are always tiny, but good practice)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"cif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
#  ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check — deployment platforms (Render, Railway) ping this."""
    return jsonify({"status": "ok"})


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    Accepts a multipart/form-data POST with:
        file     : the .cif file
        nucleus  : 'H', 'F', or 'P'     (default: 'F')
        N_wanted : integer               (default: 5000)
        B0_field : float, Tesla          (default: 9.4)
        disorder : float [0–1]           (default: 0.0)

    Returns JSON with the diffusion tensor and derived quantities.
    """

    # --- Validate file ---
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please attach a .cif file."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .cif files are accepted."}), 400

    # --- Read file content ---
    try:
        cif_text = file.read().decode("utf-8", errors="replace")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file."}), 400

    # --- Parse form parameters ---
    nucleus  = request.form.get("nucleus",  "F").strip().upper()
    disorder = float(request.form.get("disorder", 0.0))
    B0_field = float(request.form.get("B0_field", 9.4))
    N_wanted = int(request.form.get("N_wanted", 5000))

    # Clamp N_wanted to a safe range (very large N is slow in a web context)
    N_wanted = max(100, min(N_wanted, 20000))

    # --- Run calculation ---
    try:
        result = run_calculation(
            cif_text  = cif_text,
            nucleus   = nucleus,
            N_wanted  = N_wanted,
            B0_field  = B0_field,
            disorder  = disorder,
        )
        return jsonify({"success": True, "result": result})

    except ValueError as e:
        # Expected errors (bad CIF, nucleus not found, etc.)
        return jsonify({"error": str(e)}), 422

    except Exception:
        # Unexpected errors — log full traceback server-side, return safe message
        traceback.print_exc()
        return jsonify({"error": "Calculation failed due to an internal error. "
                                 "Please check your CIF file and parameters."}), 500


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # When running locally: python app.py
    # On Render: gunicorn will import this module directly (see Procfile)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
