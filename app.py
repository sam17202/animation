import torch
import os
import tempfile
import imageio
from skimage import img_as_ubyte
from skimage.transform import resize
from flask import Flask, request, send_file
from demo import load_checkpoints, make_animation

# Initialize Flask app
app = Flask(__name__)

# Model configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = "config/vox-256.yaml"
checkpoint_path = "checkpoints/vox.pth.tar"
driving_video_path = "./assets/drivevideo.mp4"
pixel = 256

# Load model once at startup
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
    config_path=config_path, checkpoint_path=checkpoint_path, device=device
)

# Load fixed driving video
reader = imageio.get_reader(driving_video_path)
fps = reader.get_meta_data()["fps"]
driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in reader]
reader.close()

@app.route("/animate", methods=["POST"])
def animate():
    try:
        if "image" not in request.files:
            return {"error": "No image uploaded"}, 400

        # Save uploaded image
        uploaded_file = request.files["image"]
        _, temp_image_path = tempfile.mkstemp(suffix=".png")
        uploaded_file.save(temp_image_path)

        # Process image
        source_image = imageio.imread(temp_image_path)
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        # Generate animation
        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode="relative",
        )

        # Save output
        _, temp_output_path = tempfile.mkstemp(suffix=".mp4")
        imageio.mimsave(temp_output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        os.remove(temp_image_path)

        return send_file(temp_output_path, mimetype="video/mp4", as_attachment=True, download_name="animation.mp4")

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Set port for Railway deployment
    app.run(host="0.0.0.0", port=port)
