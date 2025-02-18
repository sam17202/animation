!git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git

cd Thin-Plate-Spline-Motion-Model

!mkdir checkpoints
!pip3 install wldhx.yadisk-direct
!curl -L $(yadisk-direct https://disk.yandex.com/d/i08z-kCuDGLuYA) -o checkpoints/vox.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/vk5dirE6KNvEXQ) -o checkpoints/taichi.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/IVtro0k2MVHSvQ) -o checkpoints/mgif.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/B3ipFzpmkB1HIA) -o checkpoints/ted.pth.tar

import torch

# edit the config
device = torch.device('cuda:0')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
source_image_path = './assets/source.png'
driving_video_path = './assets/drivevideo.mp4'
output_video_path = './generated.mp4'
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative' # ['standard', 'relative', 'avd']
find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
if(dataset_name == 'ted'): # for ted, the resolution is 384*384
    pixel = 384

if find_best_frame:
  !pip install face_alignment

try:
  import imageio
  import imageio_ffmpeg
except:
  !pip install imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import os

warnings.filterwarnings("ignore")

source_image = imageio.imread(source_image_path)
reader = imageio.get_reader(driving_video_path)

source_image = resize(source_image, (pixel, pixel))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


HTML(display(source_image, driving_video).to_html5_video())

from demo import load_checkpoints
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

from demo import make_animation
from skimage import img_as_ubyte

if predict_mode=='relative' and find_best_frame:
    from demo import find_best_frame as _find
    i = _find(source_image, driving_video, device.type=='cpu')
    print ("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
else:
    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

#save resulting video
imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

HTML(display(source_image, driving_video, predictions).to_html5_video())

!pip install flask pyngrok imageio torch skimage

!pip install pyngrok

!pip install imageio

from flask import Flask, request, send_file
from pyngrok import ngrok
import torch
import os
import tempfile
import imageio
from skimage import img_as_ubyte
from demo import load_checkpoints, make_animation
# Import the resize function here
from skimage.transform import resize

# Initialize Flask app
app = Flask(__name__)

# Ngrok configuration
ngrok.set_auth_token("2rrH36aY2OjghIDbMHyH1e9TO8u_674w937bU9yjRn69FhMVX")  # Replace with your ngrok authtoken
public_url = ngrok.connect(5000).public_url

# Model configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
driving_video_path = './assets/drivevideo.mp4'  # Fixed driving video
pixel = 256

# Load model once at startup
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device=device
)

# Load fixed driving video
reader = imageio.get_reader(driving_video_path)
fps = reader.get_meta_data()['fps']
driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in reader] # Now resize is defined
reader.close()

@app.route('/animate', methods=['POST'])
def animate():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return {'error': 'No image uploaded'}, 400

        # Save uploaded image to temp file
        uploaded_file = request.files['image']
        _, temp_image_path = tempfile.mkstemp(suffix='.png')
        uploaded_file.save(temp_image_path)

        # Process image
        source_image = imageio.imread(temp_image_path)
        source_image = resize(source_image, (pixel, pixel))[..., :3] # resize is also defined here

        # Generate animation
        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode='relative'
        )

        # Save output to temp video file
        _, temp_output_path = tempfile.mkstemp(suffix='.mp4')
        imageio.mimsave(temp_output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        # Clean up temp image
        os.remove(temp_image_path)

        # Return generated video
        return send_file(temp_output_path, mimetype='video/mp4', as_attachment=True, download_name='animation.mp4')

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    print(f' * Public URL: {public_url}')
    app.run(host='0.0.0.0', port=5000)