from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "user_images/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = "/Users/wincheng/Desktop/VSCoding.nosync/AIFS2024/flask-server/best.pt"
model = YOLO(model_path)


@app.route("/api/upload", methods=["POST"])
def upload_and_classify():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = model(filepath)
        for result in results:
            # Assuming result.probs contains the classification probabilities
            probs = result.probs  # Probs object for classification outputs
            top1 = probs.top1

            if top1 == 0:
                top1 = "Rust"
            elif top1 == 1:
                top1 = "Mummification"
            elif top1 == 2:
                top1 = "Dot"
            elif top1 == 3:
                top1 = "Canker"

            print("TYPES", top1)

            return jsonify({"result": top1})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
