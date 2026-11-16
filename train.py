"""Reproducible CPU AdaIN fine-tuning using the existing models."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils.models import Decoder, VGGEncoder
from utils.utils import (ImageFolderDataset, adaptive_instance_normalization,
                         calc_mean_std, get_transform, image_filenames)

PROJECT = Path(__file__).resolve().parent
PRETRAINED = PROJECT / 'experiment' / 'pretrained'
RELEASE = 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/'
# Sizes and SHA-256 of the matching assets retrieved from the published release.
ASSETS = {
    'vgg_normalised.pth': (80102481, '804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a'),
    'decoder.pth': (14023458, '379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472'),
}
COCO_INDEX_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip'
COCO_INDEX_SHA256 = '51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234'
COCO_IMAGE_URL = 'https://s3.amazonaws.com/images.cocodataset.org/train2017/'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('pilot', '6000'), default='pilot',
                        help='6000 selects a longer run with 256px crops; explicit options override its defaults')
    parser.add_argument('--experiment', default='cpu_pilot')
    parser.add_argument('--content_dir', type=Path, default=PROJECT / 'content_data/train')
    parser.add_argument('--style_dir', type=Path, default=PROJECT / 'style_data/train')
    parser.add_argument('--content_train_count', type=int, default=100)
    parser.add_argument('--style_train_count', type=int, default=100)
    parser.add_argument('--val_count', type=int, default=14, help='Per collection')
    parser.add_argument('--test_count', type=int, default=14, help='Per collection')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resize_size', type=int, default=160)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max_updates', type=int, default=100, help='Total step limit, including resumed steps')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--time_budget', type=float, default=300, help='Soft training-loop seconds per invocation; 0 disables')
    parser.add_argument('--val_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=3, help='Validation checks without improvement; 0 disables')
    parser.add_argument('--min_updates', type=int, default=0, help='Do not early-stop before this many updates')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save resumable progress between validations')
    parser.add_argument('--download_workers', type=int, default=8)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--vgg_path', type=Path, default=PRETRAINED / 'vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=Path, help='Pretrained decoder weights only; no optimizer needed')
    parser.add_argument('--allow_random_decoder', action='store_true')
    parser.add_argument('--resume', type=Path, help='Explicit training checkpoint (latest.pth), not decoder weights')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check_data', action='store_true', help='Check selected images without loading models')
    modes.add_argument('--download_pretrained', action='store_true', help='Download/verify weights, then exit')
    modes.add_argument('--download_content', action='store_true', help='Download only missing COCO content photos, then exit')
    modes.add_argument('--prepare_and_train', action='store_true', help='Complete content downloads, check all selected data, then train')
    modes.add_argument('--export_deployment', type=Path, metavar='EXPERIMENT_DIR',
                       help='Export a completed experiment\'s best validated decoder for GitHub/Render')
    modes.add_argument('--prepare_deployment', action='store_true',
                       help='Verify the exported decoder and obtain only its matching VGG; never train')
    profile_args, _ = parser.parse_known_args(argv)
    if profile_args.profile == '6000':
        parser.set_defaults(experiment='adain_6000', content_train_count=6000, style_train_count=6000,
                            resize_size=512, crop_size=256, max_updates=2000, min_updates=6000,
                            val_every=500, patience=6, time_budget=0)
    args = parser.parse_args(argv)
    if not args.experiment or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.experiment):
        parser.error('--experiment must contain only letters, digits, underscores and hyphens')
    for key in ('content_train_count', 'style_train_count', 'val_count', 'batch_size',
                'threads', 'max_updates', 'val_every', 'checkpoint_every', 'download_workers'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key} must be positive')
    if args.test_count < 0 or args.patience < 0 or args.min_updates < 0:
        parser.error('--test_count, --patience and --min_updates must be nonnegative')
    if args.crop_size < 32 or args.crop_size % 8 or args.resize_size < args.crop_size:
        parser.error('--crop_size must be >=32 and divisible by 8; --resize_size must be >= --crop_size')
    for key in ('lr', 'time_budget', 'content_weight', 'style_weight', 'min_delta'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0):
            parser.error(f'Invalid --{key}')
    if args.content_weight + args.style_weight == 0:
        parser.error('At least one loss weight must be positive')
    if args.resume and (args.decoder_path or args.allow_random_decoder):
        parser.error('--resume is separate from --decoder_path / --allow_random_decoder')
    if args.decoder_path and args.allow_random_decoder:
        parser.error('Choose pretrained initialization or --allow_random_decoder, not both')
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, write):
    """Replace a complete experiment artifact, never expose a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    os.close(fd)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda temp: temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8'))


def download_content(args, directory):
    """Retrieve a seeded subset, retaining existing images and resumable download plans."""
    root = args.content_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    required = args.content_train_count + args.val_count + args.test_count
    existing = image_filenames(root)
    if len(existing) >= required:
        print(f'Content already has {len(existing)} images; {required} required. Nothing downloaded.')
        return
    index_path = PROJECT / 'experiment/coco_download/train2017_filenames.json'
    if not index_path.exists():
        # Only the filenames are retained; no annotation archive is extracted.
        print('Reading COCO filenames from the 49 MB label archive...', flush=True)
        with tempfile.TemporaryFile() as output:
            digest = hashlib.sha256()
            with urllib.request.urlopen(COCO_INDEX_URL, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
                    digest.update(chunk)
            if output.tell() != 48639045 or digest.hexdigest() != COCO_INDEX_SHA256:
                raise ValueError('COCO index archive size/SHA-256 mismatch')
            output.seek(0)
            with zipfile.ZipFile(output) as archive:
                members = [name for name in archive.namelist() if name.endswith('train2017.txt')]
                if len(members) != 1:
                    raise ValueError('COCO training filename list missing or ambiguous')
                names = sorted({Path(line).name for line in archive.read(members[0]).decode('utf-8').splitlines() if line.strip()})
            write_json(index_path, {'source': COCO_INDEX_URL, 'archive_sha256': digest.hexdigest(), 'filenames': names})
    index = json.loads(index_path.read_text(encoding='utf-8'))
    names = index['filenames']
    if len(set(names)) != 118287 or any(len(name) != 16 or not name[:12].isdigit() or name[12:] != '.jpg' for name in names):
        raise ValueError('Invalid COCO 2017 filename index')
    plan_path = directory / 'content_download.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if (plan['root'], plan['required'], plan['seed']) != (str(root), required, args.seed):
            raise ValueError('Existing content download plan differs; use its settings or a new experiment')
    else:
        available = set(existing)
        candidates = [name for name in names if name not in available]
        random.Random(args.seed).shuffle(candidates)
        if len(candidates) < required - len(existing):
            raise ValueError('Not enough new COCO training images for the requested count')
        plan = dict(root=str(root), required=required, seed=args.seed, original_files=existing,
                    source=COCO_IMAGE_URL, index_source=index['source'],
                    filenames=candidates[:required - len(existing)], completed={})
        write_json(plan_path, plan)

    def fetch(name):
        path = root / name
        if path.exists():
            with Image.open(path) as picture:
                picture.load()
            return name, {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for attempt in range(3):
            fd, temporary = tempfile.mkstemp(dir=root, prefix=name + '.', suffix='.part')
            os.close(fd)
            temp = Path(temporary)
            try:
                with urllib.request.urlopen(COCO_IMAGE_URL + name, timeout=45) as response, temp.open('wb') as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(chunk)
                with Image.open(temp) as picture:
                    picture.load()
                    if picture.format != 'JPEG':
                        raise ValueError(f'{name}: expected a COCO JPEG image')
                details = {'sha256': sha256(temp), 'bytes': temp.stat().st_size}
                os.link(temp, path)  # Exclusive publication; never overwrite a user's image.
                return name, details
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
            finally:
                temp.unlink(missing_ok=True)

    print(f'Downloading/verifying {len(plan["filenames"])} selected COCO photos; existing images are retained.', flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        pending = {pool.submit(fetch, name): name for name in plan['filenames']}
        for number, future in enumerate(as_completed(pending), start=1):
            try:
                name, details = future.result()
                plan['completed'][name] = details
            except Exception as exc:
                failures.append(f'{pending[future]}: {exc}')
            if number % 100 == 0 or number == len(pending):
                plan['failures'] = failures
                write_json(plan_path, plan)
                print(f'COCO progress: {number}/{len(pending)} checked, {len(failures)} failures', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} COCO downloads failed. Rerun --download_content to retry. First: {failures[0]}')
    print(f'Content download complete: {len(image_filenames(root))} available images.', flush=True)


def download_pretrained(args, include_decoder=True):
    """Verify pinned release bytes and architecture before publishing a new file."""
    destinations = {'vgg_normalised.pth': args.vgg_path}
    if include_decoder:
        destinations['decoder.pth'] = args.decoder_path or PRETRAINED / 'decoder.pth'
    for name, path in destinations.items():
        expected_size, expected_hash = ASSETS[name]
        validate = VGGEncoder if name == 'vgg_normalised.pth' else lambda p: Decoder().load_weights(p)
        if path.exists():
            validate(path)
            print(f'Kept existing compatible weights: {path} (SHA-256 {sha256(path)})')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=name + '.', suffix='.part')
        os.close(fd)
        temp = Path(temporary)
        try:
            print(f'Downloading {RELEASE + name}', flush=True)
            request = urllib.request.Request(RELEASE + name, headers={'User-Agent': 'AdaIN-CPU-pilot'})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
            if temp.stat().st_size != expected_size or sha256(temp) != expected_hash:
                raise ValueError(f'{name}: size/SHA-256 mismatch; download rejected')
            validate(temp)
            # Hard-link creation is exclusive: even a concurrent download cannot overwrite.
            os.link(temp, path)
            print(f'Verified and saved {path} (SHA-256 {expected_hash})')
        finally:
            temp.unlink(missing_ok=True)


def export_deployment(source):
    source = source.resolve()
    selection = json.loads((source / 'selection.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(source / 'latest.pth', map_location='cpu', weights_only=True)
    model = Decoder().load_weights(source / 'decoder_final.pth')
    if (checkpoint['step'] != summary['latest_step'] or checkpoint['step'] != selection['latest_step']
            or checkpoint['best_step'] != selection['best_step']
            or checkpoint['best_loss'] != selection['best_validation_loss']):
        raise ValueError('Checkpoint and selection metadata differ; wait for training to finish before exporting')
    if any(not torch.equal(value, checkpoint['best_decoder'][key]) for key, value in model.state_dict().items()):
        raise ValueError('decoder_final.pth is not the best decoder recorded in latest.pth')
    target = PROJECT / 'experiment/deployment'
    atomic_write(target / 'decoder_final.pth', lambda temporary: shutil.copyfile(source / 'decoder_final.pth', temporary))
    write_json(target / 'model_info.json', {
        'source_experiment': source.name,
        'best_step': selection['best_step'], 'completed_updates': checkpoint['step'],
        'best_validation_loss': selection['best_validation_loss'],
        'content_train_count': len(checkpoint['manifest']['collections']['content']['train']),
        'style_train_count': len(checkpoint['manifest']['collections']['style']['train']),
        'decoder_sha256': sha256(target / 'decoder_final.pth'),
        'vgg_sha256': checkpoint['config']['vgg_sha256'],
    })
    print(f'Exported validated step {selection["best_step"]} to {target}. No dataset or optimizer state included.')


def prepare_deployment(args):
    target = PROJECT / 'experiment/deployment'
    metadata = json.loads((target / 'model_info.json').read_text(encoding='utf-8'))
    path = target / 'decoder_final.pth'
    if sha256(path) != metadata['decoder_sha256']:
        raise ValueError('Deployment decoder SHA-256 mismatch; export and commit both deployment files together')
    Decoder().load_weights(path)  # Fail if missing/incompatible; never substitute the release decoder.
    download_pretrained(args, include_decoder=False)
    if sha256(args.vgg_path) != metadata['vgg_sha256']:
        raise ValueError('VGG does not match the encoder used for the exported decoder')
    print(f'Deployment weights verified: {metadata["source_experiment"]}, selected step {metadata["best_step"]}.')


def select_data(args, directory):
    path = directory / 'selected_files.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if previous and previous['seed'] != args.seed:
        raise ValueError(f'{path}: seed differs; use a new --experiment')
    manifest = {'version': 1, 'seed': args.seed, 'collections': {}}
    for offset, name in enumerate(('content', 'style')):
        root = getattr(args, name + '_dir').resolve()
        files = image_filenames(root)
        counts = (getattr(args, name + '_train_count'), args.val_count, args.test_count)
        required = sum(counts)
        print(f'{name}: {len(files)} available; requested train={counts[0]}, validation={counts[1]}, test={counts[2]}')
        if len(files) < required:
            raise ValueError(f'{root}: need {required} distinct images ({counts[0]} train + '
                             f'{counts[1]} validation + {counts[2]} test), found {len(files)}. '
                             'Add images or reduce the counts; splits will not overlap.')
        if previous:
            saved = previous['collections'][name]
            selected = [saved[key] for key in ('train', 'validation', 'test')]
            flat = [filename for split in selected for filename in split]
            if (saved['root'] != str(root) or tuple(map(len, selected)) != counts
                    or len(set(flat)) != len(flat) or not set(flat).issubset(files)):
                raise ValueError(f'{path}: folders/counts/selected files differ; use a new --experiment')
            # Additional files must not reshuffle an established experiment or its holdouts.
            manifest['collections'][name] = saved
            continue
        random.Random(args.seed + offset).shuffle(files)
        train_end, val_end = counts[0], counts[0] + counts[1]
        manifest['collections'][name] = {
            'root': str(root), 'available_count': len(files),
            'train': files[:train_end], 'validation': files[train_end:val_end],
            'test': files[val_end:required],
        }
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError(f'{path} differs from this selection. Use a new --experiment to preserve the old split.')
    else:
        write_json(path, manifest)
    return manifest


def make_datasets(args, manifest):
    return {
        (name, split): ImageFolderDataset(
            collection['root'],
            get_transform(args.resize_size, True, args.crop_size, deterministic=split != 'train'),
            collection[split],
        )
        for name, collection in manifest['collections'].items()
        for split in ('train', 'validation', 'test')
    }


def check_data(args, datasets):
    # Read every selected image, including holdouts; test pixels are never used by training.
    for (name, split), dataset in datasets.items():
        batches = 0
        seen = 0
        shapes = set()
        for batch in DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False):
            if tuple(batch.shape[1:]) != (3, args.crop_size, args.crop_size) or not torch.isfinite(batch).all():
                raise ValueError(f'{name}/{split}: unexpected/non-finite tensor {tuple(batch.shape)}')
            batches += 1
            seen += len(batch)
            shapes.add(tuple(batch.shape))
        print(f'Checked {name}/{split}: {seen} images, {batches} batches, tensor shapes={sorted(shapes)}')
    print('Data check passed. No model weights loaded and no training started.')


class CyclingBatchSampler:
    """Independent shuffled cycles, full batches, and resumable sampling position."""
    def __init__(self, count, batch_size, seed):
        self.count = count
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order = []
        self.position = 0

    def __iter__(self):
        while True:
            batch = []
            for _ in range(self.batch_size):
                if self.position == len(self.order):
                    self.order = list(range(self.count))
                    self.rng.shuffle(self.order)
                    self.position = 0
                batch.append(self.order[self.position])
                self.position += 1
            yield batch

    def state_dict(self):
        return {'order': self.order[:], 'position': self.position, 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['order'] and sorted(state['order']) != list(range(self.count)):
            raise ValueError('Checkpoint sampler does not match the selected dataset')
        if not 0 <= state['position'] <= len(state['order']):
            raise ValueError('Invalid checkpoint sampler position')
        self.order = state['order'][:]
        self.position = state['position']
        self.rng.setstate(state['rng'])


def losses(encoder, decoder, content, style):
    with torch.no_grad():
        content_features = encoder(content, is_test=True)
        style_features = encoder(style)
        target = adaptive_instance_normalization(content_features, style_features[-1])
    generated = decoder(target)
    # Frozen parameters still allow differentiation with respect to generated pixels.
    generated_features = encoder(generated)
    content_loss = F.mse_loss(generated_features[-1], target)
    style_loss = generated.new_zeros(())
    for actual, expected in zip(generated_features, style_features):
        actual_mean, actual_std = calc_mean_std(actual)
        expected_mean, expected_std = calc_mean_std(expected)
        style_loss = style_loss + F.mse_loss(actual_mean, expected_mean) + F.mse_loss(actual_std, expected_std)
    return content_loss, style_loss, generated


def check_gradients(encoder, decoder):
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError('Frozen VGG unexpectedly received gradients')
    gradients = [parameter.grad for parameter in decoder.parameters()]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError('Decoder gradients are missing or non-finite; update rejected')
    if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise RuntimeError('All decoder gradients are zero; update rejected')


@torch.no_grad()
def validate(args, encoder, decoder, datasets):
    decoder.eval()
    content = datasets['content', 'validation']
    style = datasets['style', 'validation']
    totals = [0.0, 0.0, 0.0]
    preview = None
    processed = 0
    for index in range(max(len(content), len(style))):
        c = content[index % len(content)].unsqueeze(0)
        s = style[index % len(style)].unsqueeze(0)
        loss_c, loss_s, generated = losses(encoder, decoder, c, s)
        loss = args.content_weight * loss_c + args.style_weight * loss_s
        values = (loss.item(), loss_c.item(), loss_s.item())
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Non-finite validation loss')
        totals = [total + value for total, value in zip(totals, values)]
        processed += 1
        if preview is None:
            preview = (c, s, generated.clamp(0, 1))
    return [total / processed for total in totals], preview


def save_previews(directory, label, preview):
    folder = directory / 'previews'
    folder.mkdir(exist_ok=True)
    for name, tensor in zip(('content', 'style', 'output'), preview):
        save_image(tensor, folder / f'{label}_{name}.png')


def clone_weights(decoder):
    return {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}


def run_training(args, directory, manifest, datasets):
    if not args.vgg_path.is_file():
        raise ValueError(f'Missing normalized VGG: {args.vgg_path}. Run python train.py --download_pretrained.')
    checkpoint_path = directory / 'latest.pth'
    if not args.resume and any((directory / name).exists() for name in ('latest.pth', 'decoder_final.pth', 'losses.csv')):
        raise ValueError('Experiment already has training outputs. Use --resume or choose a new --experiment.')
    encoder = VGGEncoder(args.vgg_path).cpu().eval()
    decoder = Decoder().cpu()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    # Changes to these would invalidate continuation or loss-based model selection.
    config = {key: getattr(args, key) for key in (
        'seed', 'resize_size', 'crop_size', 'batch_size', 'threads', 'lr',
        'content_weight', 'style_weight', 'min_delta', 'val_every', 'patience', 'min_updates')}
    config['vgg_sha256'] = sha256(args.vgg_path)
    samplers = {name: CyclingBatchSampler(len(datasets[name, 'train']), args.batch_size, args.seed + offset + 100)
                for offset, name in enumerate(('content', 'style'))}
    iterators = {name: iter(DataLoader(datasets[name, 'train'], batch_sampler=sampler,
                                     num_workers=0, pin_memory=False,
                                     generator=torch.Generator().manual_seed(args.seed)))
                 for name, sampler in samplers.items()}
    step = 0
    best_step = 0
    bad_checks = 0
    last_val_step = 0
    history = []
    train_totals = [0.0, 0.0, 0.0]
    train_batches = 0
    update_seconds = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('--resume requires a latest.pth training checkpoint, not standalone decoder weights')
        checkpoint['config'].setdefault('min_updates', 0)  # Compatibility with existing pilot checkpoints.
        if checkpoint['config'] != config or checkpoint['manifest'] != manifest:
            raise ValueError('Resume configuration/split differs. Reuse the original settings and selected files.')
        decoder.load_state_dict(checkpoint['decoder'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_state = checkpoint['best_decoder']
        Decoder().load_state_dict(best_state, strict=True)
        for state in (checkpoint['decoder'], best_state):
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError('Resume decoder contains non-finite weights')
        step = checkpoint['step']
        best_step = checkpoint['best_step']
        best_loss = checkpoint['best_loss']
        bad_checks = checkpoint['bad_checks']
        last_val_step = checkpoint['last_val_step']
        history = checkpoint['history']
        train_totals = checkpoint['train_totals']
        train_batches = checkpoint['train_batches']
        update_seconds = checkpoint['update_seconds']
        for name, sampler in samplers.items():
            sampler.load_state_dict(checkpoint['samplers'][name])
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        print(f'Resumed step {step}; best validated step {best_step}, loss {best_loss:.6f}')
    else:
        if args.allow_random_decoder:
            print('Explicit decoder-from-scratch experiment: useful quality is unlikely in this tiny CPU pilot.')
        else:
            path = args.decoder_path or PRETRAINED / 'decoder.pth'
            if not path.is_file():
                raise ValueError(f'Missing pretrained decoder: {path}. Run python train.py --download_pretrained '
                                 'or supply --decoder_path. Random initialization requires --allow_random_decoder.')
            decoder.load_weights(path)
        started = time.perf_counter()
        values, preview = validate(args, encoder, decoder, datasets)
        best_loss = values[0]
        best_state = clone_weights(decoder)
        history.append(dict(kind='validation', step=0, loss=values[0], content_loss=values[1],
                            style_loss=values[2], seconds=time.perf_counter() - started))
        save_previews(directory, 'initial', preview)
        save_previews(directory, 'best', preview)
        print(f'Initial validation: loss={best_loss:.6f} ({history[-1]["seconds"]:.3f}s)')

    def save_progress():
        # Latest decoder, Adam state, sampler position and RNG all describe this step.
        checkpoint = dict(
            format_version=1, config=config, manifest=manifest, step=step,
            decoder=clone_weights(decoder), optimizer=optimizer.state_dict(),
            best_decoder=best_state, best_step=best_step, best_loss=best_loss,
            bad_checks=bad_checks, last_val_step=last_val_step, history=history,
            train_totals=train_totals, train_batches=train_batches, update_seconds=update_seconds,
            samplers={name: sampler.state_dict() for name, sampler in samplers.items()},
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
        )
        atomic_write(checkpoint_path, lambda temp: torch.save(checkpoint, temp))
        atomic_write(directory / 'decoder_final.pth', lambda temp: torch.save(best_state, temp))
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=['kind', 'step', 'loss', 'content_loss', 'style_loss', 'seconds'])
        writer.writeheader()
        writer.writerows(history)
        atomic_write(directory / 'losses.csv', lambda temp: temp.write_text(output.getvalue(), encoding='utf-8'))
        write_json(directory / 'selection.json', {'best_step': best_step, 'best_validation_loss': best_loss,
                                                'latest_step': step, 'last_validation_step': last_val_step})

    def validation_check():
        nonlocal best_loss, best_step, best_state, bad_checks, last_val_step
        started = time.perf_counter()
        values, preview = validate(args, encoder, decoder, datasets)
        last_val_step = step
        history.append(dict(kind='validation', step=step, loss=values[0], content_loss=values[1],
                            style_loss=values[2], seconds=time.perf_counter() - started))
        significant_improvement = values[0] < best_loss - args.min_delta
        if values[0] < best_loss:
            best_loss, best_step, best_state = values[0], step, clone_weights(decoder)
            save_previews(directory, 'best', preview)
        bad_checks = 0 if significant_improvement else bad_checks + 1
        save_previews(directory, 'latest_validation', preview)
        print(f'Validation step {step}: loss={values[0]:.6f}; best={best_loss:.6f} at step {best_step}; '
              f'checks without improvement={bad_checks}', flush=True)
        save_progress()

    write_json(directory / 'config.json', {**{key: str(value) if isinstance(value, Path) else value
                                            for key, value in vars(args).items()}, 'verified_config': config})
    save_progress()  # Baseline survives even if the first update never finishes.
    budget_label = f'{args.time_budget:g}s' if args.time_budget else 'disabled'
    print(f'CPU threads={args.threads}; num_workers=0; pin_memory=False. '
          f'Soft loop budget={budget_label}, total update limit={args.max_updates}.\n'
          'Model loading, initial validation and saving add time. An in-flight update/validation can overrun the budget.',
          flush=True)
    loop_started = time.perf_counter()
    def within_budget():
        return args.time_budget == 0 or time.perf_counter() - loop_started < args.time_budget

    stop_reason = 'maximum updates'
    while step < args.max_updates:
        if args.patience and step >= args.min_updates and bad_checks >= args.patience:
            stop_reason = 'early stopping'
            break
        if (directory / 'stop_requested').exists():
            stop_reason = 'stop requested'
            break
        if not within_budget():
            stop_reason = 'soft time budget'
            break
        started = time.perf_counter()
        content = next(iterators['content'])
        style = next(iterators['style'])
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        loss_c, loss_s, _ = losses(encoder, decoder, content, style)
        loss = args.content_weight * loss_c + args.style_weight * loss_s
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; update rejected')
        loss.backward()
        check_gradients(encoder, decoder)
        optimizer.step()
        step += 1
        elapsed = time.perf_counter() - started
        update_seconds += elapsed
        values = (loss.item(), loss_c.item(), loss_s.item())
        train_totals = [total + value for total, value in zip(train_totals, values)]
        train_batches += 1
        history.append(dict(kind='train', step=step, loss=values[0], content_loss=values[1],
                            style_loss=values[2], seconds=elapsed))
        print(f'Update {step}/{args.max_updates}: loss={values[0]:.6f}, measured {elapsed:.3f}s/update; '
              f'mean={update_seconds / train_batches:.3f}s/update; '
              f'content passes={step * args.batch_size / len(datasets["content", "train"]):.3f}, '
              f'style passes={step * args.batch_size / len(datasets["style", "train"]):.3f}', flush=True)
        if step % args.val_every == 0 and within_budget():
            validation_check()
        elif step % args.checkpoint_every == 0:
            save_progress()
    if step != last_val_step and within_budget() and stop_reason != 'stop requested':
        validation_check()
    elif step != last_val_step:
        print('Stopped before validation: unvalidated updates remain only in latest.pth; decoder_final.pth keeps the validated best.')
    loop_seconds = time.perf_counter() - loop_started
    save_progress()
    # Check the actual exported artifact using the same strict path as the Flask app.
    restored = Decoder().load_weights(directory / 'decoder_final.pth')
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in best_state.items()):
        raise RuntimeError('Exported decoder failed its round-trip check')
    averages = [total / train_batches for total in train_totals] if train_batches else None
    write_json(directory / 'summary.json', dict(
        stop_reason=stop_reason, latest_step=step, best_step=best_step, best_validation_loss=best_loss,
        processed_training_batches=train_batches, mean_training_losses=averages,
        measured_mean_update_seconds=update_seconds / train_batches if train_batches else None,
        loop_seconds_this_invocation=loop_seconds, test_images_used=False,
    ))
    if (stop_reason in ('maximum updates', 'early stopping')
            and step * args.batch_size >= max(len(datasets['content', 'train']), len(datasets['style', 'train']))):
        write_json(directory / 'ready_for_app.json', {
            'decoder': 'decoder_final.pth', 'sha256': sha256(directory / 'decoder_final.pth'),
            'latest_step': step, 'best_step': best_step, 'best_validation_loss': best_loss,
            'content_train_count': len(datasets['content', 'train']),
            'style_train_count': len(datasets['style', 'train']),
        })
    print(f'Stopped: {stop_reason}. Best decoder: {directory / "decoder_final.pth"} (step {best_step}).\n'
          f'Mean training losses over {train_batches} actual batches: {averages}.\n'
          'Lower loss does not automatically mean better-looking images; compare the fixed previews.')


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.export_deployment:
        export_deployment(args.export_deployment)
        return
    if args.prepare_deployment:
        prepare_deployment(args)
        return
    if args.download_pretrained:
        download_pretrained(args)
        return
    directory = PROJECT / 'experiment' / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    if args.download_content:
        download_content(args, directory)
        return
    def status(stage, **extra):
        if args.prepare_and_train:
            write_json(directory / 'run_status.json', dict(stage=stage, pid=os.getpid(), **extra))

    try:
        if args.prepare_and_train:
            status('downloading_content')
            download_content(args, directory)
        status('checking_data')
        manifest = select_data(args, directory)
        datasets = make_datasets(args, manifest)
        if args.check_data or args.prepare_and_train:
            check_data(args, datasets)
            if args.check_data:
                return
        status('training')
        run_training(args, directory, manifest, datasets)
        status('finished', summary=json.loads((directory / 'summary.json').read_text(encoding='utf-8')))
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
