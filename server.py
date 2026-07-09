from flask import Flask, render_template, send_from_directory
import sqlite3

app = Flask(__name__)

@app.route("/alerts/<path:filename>")
def alerts(filename):
    return send_from_directory("alerts", filename)

@app.route("/")
def dashboard():
    conn = sqlite3.connect("surveillance.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    detections = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM detections")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE name='Unknown'")
    unknown = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE name!='Unknown'")
    known = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE name LIKE 'Blacklisted%'")
    blacklisted = cursor.fetchone()[0]

    conn.close()

    return render_template(
    "dashboard.html",
    detections=detections,
    total=total,
    known=known,
    unknown=unknown,
    blacklisted=blacklisted
)

if __name__ == "__main__":
    app.run(debug=True)