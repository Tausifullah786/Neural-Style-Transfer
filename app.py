import os
import json
import secrets
import uuid
from pathlib import Path
import torch
from flask import Flask, abort, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired, NumberRange
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


PROJECT_ROOT = Path(__file__).resolve().parent
IS_PRODUCTION = os.environ.get('NST_ENV') == 'production' or os.environ.get('RENDER') == 'true'
secret_key = os.environ.get('SECRET_KEY')
if IS_PRODUCTION and not secret_key:
    raise RuntimeError('Set SECRET_KEY in the deployment environment before starting the app')

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key or secrets.token_hex(32)
upload_folder = Path(os.environ.get('NST_UPLOAD_DIR', 'static/uploads'))
if not upload_folder.is_absolute():
    upload_folder = PROJECT_ROOT / upload_folder
app.config['UPLOAD_FOLDER'] = str(upload_folder)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0, validators=[InputRequired(), NumberRange(min=0, max=1)])
    submit = SubmitField('Transfer Style')

GALLERY_ASSET_DIR = PROJECT_ROOT / 'static' / 'gallery'
GALLERY_ASSETS = {
    'river-content.webp', 'river-style.webp', 'river-result.webp',
    'alpine-content.webp', 'alpine-style.webp', 'alpine-result.webp',
    'wildlife-content.webp', 'wildlife-style.webp', 'wildlife-result.webp',
}

def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


device = torch.device('cpu')
torch.set_num_threads(int(os.environ.get('NST_CPU_THREADS', '2')))
encoder_path = project_path(os.environ.get('NST_VGG_PATH', 'experiment/pretrained/vgg_normalised.pth'))


def selected_decoder_path():
    override = os.environ.get('NST_DECODER_PATH')
    if override:
        return project_path(override)
    if IS_PRODUCTION:
        return PROJECT_ROOT / 'experiment/deployment/decoder_final.pth'
    candidate = PROJECT_ROOT / 'experiment/adain_6000'
    marker = candidate / 'ready_for_app.json'
    if marker.is_file():
        selection = json.loads(marker.read_text(encoding='utf-8'))
        if selection['content_train_count'] >= 6000 and selection['style_train_count'] >= 6000:
            return candidate / 'decoder_final.pth'
    return PROJECT_ROOT / 'experiment/cpu_pilot/decoder_final.pth'


decoder_path = selected_decoder_path()
encoder = VGGEncoder(encoder_path).to(device)
decoder = Decoder().load_weights(decoder_path).to(device)
decoder_version = (decoder_path, decoder_path.stat().st_mtime_ns)

encoder.eval()
decoder.eval()


@app.before_request
def refresh_selected_decoder():
    """Adopt the completed large experiment on the next request, or honor an explicit path."""
    global decoder, decoder_path, decoder_version
    selected = selected_decoder_path()
    version = (selected, selected.stat().st_mtime_ns)
    if version != decoder_version:
        replacement = Decoder().load_weights(selected).to(device).eval()
        decoder, decoder_path, decoder_version = replacement, selected, version
        app.logger.info('Loaded validated decoder: %s', selected)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    if not 0 <= alpha <= 1:
        raise ValueError('Style Strength must be between 0 and 1')
    content_transform = transforms.Compose([
        transforms.Resize(512, max_size=1024, antialias=True),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512, max_size=1024, antialias=True),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def read_upload(filename):
    if not filename or secure_filename(filename) != filename or not allowed_file(filename):
        raise ValueError('Invalid uploaded image filename')
    with Image.open(Path(app.config['UPLOAD_FOLDER']) / filename) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError('Please use an image with at most 16 million pixels')
        if max(source.size) > 32 * min(source.size):
            raise ValueError('Please use an image with an aspect ratio of at most 32:1')
        return source.convert('RGB')


@app.route('/healthz')
def health():
    return {'status': 'ok', 'device': device.type}



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = uuid.uuid4().hex + Path(secure_filename(form.content.data.filename)).suffix.lower()
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = uuid.uuid4().hex + Path(secure_filename(form.style.data.filename)).suffix.lower()
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            try:
                content_image = read_upload(content_filename)
                style_image = read_upload(style_filename)

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + Path(content_filename).stem + '.png'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'
        if form.alpha.errors:
            error = 'Style Strength must be between 0 and 1'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    if filename not in GALLERY_ASSETS:
        abort(404)
    return send_from_directory(GALLERY_ASSET_DIR, filename, mimetype='image/webp')


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=not IS_PRODUCTION, use_debugger=not IS_PRODUCTION)
