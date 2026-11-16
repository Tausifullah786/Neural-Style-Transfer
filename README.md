# Neural-Style-Transfer

## Local setup (Windows PowerShell)

Use Python 3.12 for the pinned NumPy and PyTorch dependencies. Python 3.14 is
incompatible with these versions.

```powershell
py -3.12 -m venv myenv
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

In your IDE, select `myenv\Scripts\python.exe` as the Python interpreter.
To activate the environment in a new PowerShell terminal:

```powershell
.\myenv\Scripts\Activate.ps1
```

Use `myenv` as the single local project environment. Install packages through
`.\myenv\Scripts\python.exe -m pip` so they go into this environment.
`requirements.txt` pins the installed package versions, including their current
dependencies. Check consistency after dependency changes:

```powershell
.\myenv\Scripts\python.exe -m pip check
```

## Deployment

The repository now includes a Render Blueprint in `render.yaml`. It uses
Python 3.12 (from `.python-version`), CPU-only PyTorch, one Gunicorn worker,
an automatically generated secret key, and `/healthz` for health checks.

The deployed model is an explicit snapshot in `experiment/deployment/`:

- `decoder_final.pth`: approximately 14 MB, exported from the best validated
  step **500** of `adain_6000`.
- `model_info.json`: selection details and decoder/VGG SHA-256 hashes.

The local run completed **2,000 updates**, approximately one-third of one pass
through the selected 6,000 images per collection at batch size 1. It did not
complete the original 12,000-update target. The export represents its best
validated checkpoint; visual improvement is not guaranteed.

`.gitignore` excludes `myenv`, both datasets, uploaded/generated user images,
secrets, logs and training checkpoints. Only the explicit deployment snapshot
is allowed through under `experiment/`. Gallery assets remain included. The
80 MB normalized VGG downloads during the Render build and is checked against
the encoder hash recorded with the selected decoder. The build fails if the
exported decoder is missing or mismatched; it never substitutes a different
decoder or starts training.

### Push to your existing GitHub repository

From PowerShell, run these commands in order:

```powershell
Set-Location 'D:\Neural-Style-Transfer'
.\myenv\Scripts\python.exe train.py --prepare_deployment
git status --short
git add .gitignore .python-version README.md app.py train.py requirements.txt render.yaml utils templates static/gallery experiment/deployment
git diff --cached --stat
git commit -m "Prepare AdaIN app and validated model for Render"
git push -u origin main
```

The existing remote is
`https://github.com/Tausifullah786/Neural-Style-Transfer.git` and the branch is
`main`; no `git init` or `git remote add` is needed. Local and remote HEAD were
equal when these instructions were prepared. If remote commits are added
before you push, run `git pull --rebase origin main`, resolve any reported
conflicts, then retry the normal push. Do not force-push over remote changes.

### Create the Render service

In Render, choose **New > Blueprint**, connect the GitHub repository, choose
`main`, and review the settings from `render.yaml` before creating the service.
The Blueprint selects a **paid 1-CPU / 2-GB instance (`1c-2g`)**. Check its current
price in Render before deploying. The local 1024-by-512 inference smoke test
peaked at approximately **940 MiB**, above Render's free 512 MB limit. This is
a Windows measurement, not a guarantee of Linux memory usage or latency.

Render will use these commands automatically:

```sh
# Build
python -m pip install --no-cache-dir -r requirements.txt && python train.py --prepare_deployment

# Start (Linux/Render; PORT is supplied by Render)
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 --access-logfile - --error-logfile -
```

The Blueprint sets `NST_ENV=production`, `NST_CPU_THREADS=1`,
`NST_DECODER_PATH=experiment/deployment/decoder_final.pth`, and the normalized
VGG path. It generates `SECRET_KEY`; production startup refuses to continue
without that environment variable. Do not commit a real secret key.
Python's minor version in `.python-version` makes Render select a supported
3.12 patch release rather than its newer default Python version.

The app accepts up to 12 MB per request and 16 million pixels per image. It
caps the resized longest edge at 1024 pixels, assigns unique upload filenames,
validates reused image paths, and saves results as PNG. Render uploads go to
`/tmp/neural-style-transfer-uploads`; they are temporary and may disappear
on restart or redeploy. Download any results you want to keep.

After deployment, check `/healthz`, load the home page and submit a real
content/style pair. Local checks passed for production configuration, signed
form submission, model loading, image output, all gallery assets and the health
endpoint. PyPI version availability and the official Linux/Python-3.12 CPU
PyTorch wheels were checked. **The actual Render build and Gunicorn deployment
have not been run from this workspace.** Render logs remain the final check.

### Publish a later trained model

Once another run finishes, explicitly export its selected checkpoint, then
commit and push the two snapshot files:

```powershell
.\myenv\Scripts\python.exe train.py --export_deployment experiment/adain_6000
.\myenv\Scripts\python.exe train.py --prepare_deployment
git add experiment/deployment/decoder_final.pth experiment/deployment/model_info.json
git commit -m "Update validated deployment decoder"
git push
```

The export cross-checks the standalone decoder against the best weights and
selection metadata in the resumable checkpoint. It does not modify the training
experiment. Render is configured to redeploy commits to `main` once connected.

References: [Render Flask deployment](https://render.com/docs/deploy-flask),
[Blueprint settings](https://render.com/docs/blueprint-spec), and
[Python version selection](https://render.com/docs/python-version).

## Small CPU AdaIN pilot

The existing VGGEncoder and Decoder architectures and AdaIN mean/standard-deviation
calculation are retained. This checkout did not contain `train.py`; the training
entry point now uses those existing modules. The normalized VGG stays frozen.
Only the decoder is fine-tuned. A tiny dataset is a pilot, and useful results with
little CPU training depend on suitable pretrained weights. Lower loss does not
automatically mean better-looking images: compare the saved previews.

Keep photographs directly inside `content_data/train/` and paintings directly
inside `style_data/train/`. Files are sorted before seeded selection. JPG, JPEG
and PNG are accepted regardless of extension case; directories and other files
are ignored. Images are opened with closed file handles and converted to RGB.
Unreadable selected files produce an error containing their path.

Run these commands **in order**, from this project directory, in PowerShell:

```powershell
# 1. Verify all selected train/validation/test images and tensor shapes.
.\myenv\Scripts\python.exe train.py --check_data --experiment cpu_pilot

# 2. Obtain missing pretrained weights (exits without training).
.\myenv\Scripts\python.exe train.py --download_pretrained

# 3. Run the short CPU fine-tuning experiment.
.\myenv\Scripts\python.exe train.py --experiment cpu_pilot --decoder_path experiment/pretrained/decoder.pth --threads 2

# 4. Launch the existing Flask app with the selected best decoder, on CPU.
$env:NST_DECODER_PATH = 'experiment/cpu_pilot/decoder_final.pth'
.\myenv\Scripts\python.exe app.py
```

Open http://localhost:5000 after launching. The app renders the existing
`templates/index.html`. Routes and form fields are preserved. Generated images
are saved as lossless PNGs, and decorative scanlines sit behind the content
cards so they no longer draw horizontal lines across the result.

The two matching assets come from the published
[PyTorch AdaIN v0.0.0 release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0).
Downloads go to `experiment/pretrained/`, use HTTPS, and are checked against
pinned byte sizes and SHA-256 hashes plus strict model keys/shapes before being
published. The release's API does not supply digests for these older assets;
the hashes below were computed from the matching assets retrieved over HTTPS:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `vgg_normalised.pth` | 80102481 | `804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a` |
| `decoder.pth` | 14023458 | `379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472` |

Existing files are inspected for compatibility and retained; an incompatible
existing file causes an error rather than being overwritten. Custom destinations
can be supplied with `--vgg_path` and `--decoder_path` on the download command.
Ordinary torchvision VGG weights are incompatible with this normalized VGG.
Weights are read on CPU with `weights_only=True`; no unsafe pickle fallback is
used. Decoders may contain `net.1.weight`-style keys or the release's bare
Sequential `1.weight`-style keys. Missing or mismatched weights are fatal.
`--decoder_path` loads pretrained decoder weights without an optimizer or
`--resume`. Training a random decoder requires `--allow_random_decoder` explicitly.

### Defaults and configuration

| Option | Default | Meaning |
| --- | ---: | --- |
| `--content_train_count` | 100 | Selected content training images |
| `--style_train_count` | 100 | Selected style training images |
| `--val_count` | 14 | Validation images from **each** collection |
| `--test_count` | 14 | Reserved test images from **each** collection |
| `--seed` | 42 | Reproducible splits, ordering and training crops |
| `--resize_size` | 160 | Resize the shorter edge, keeping aspect ratio |
| `--crop_size` | 128 | Final square crop; at least 32 and divisible by 8 |
| `--batch_size` | 1 | Training batch size |
| `--threads` | 2 | PyTorch CPU computation threads; inter-op threads = 1 |
| `--max_updates` | 100 | Total optimizer updates, including resumed steps |
| `--lr` | 0.00001 | Adam fine-tuning learning rate |
| `--time_budget` | 300 | Soft training-loop seconds per invocation |
| `--val_every` | 25 | Updates between fixed validation checks |
| `--patience` | 3 | Checks without improvement before stopping; 0 disables |
| `--min_delta` | 0 | Required validation improvement for early stopping |
| `--content_weight` / `--style_weight` | 1 / 10 | Content/style loss weights |

Use `--content_dir` and `--style_dir` to change the input directories without
moving images. Each collection needs train + validation + test distinct images
(128 by default). Insufficient data is an error, never an overlapping split.
Selections are saved in `experiment/<experiment>/selected_files.json`. Repeating
an experiment reuses that exact selection even if additional images are added;
changed counts, seed, folders or missing selected files require a new experiment
name. The recorded available count describes the inventory when the split was
first created. Test images are read only by
`--check_data`; they do not enter training, validation, previews or selection.

Training uses random crops after resizing; validation/test checks use fixed
center crops. Every training batch has a consistent shape. Independent shuffled
content/style cycles fill complete batches across cycle boundaries, so a shorter
collection never truncates a longer one. Training averages divide by the actual
processed batch count. Validation averages divide by its actual image-pair count.
DataLoaders use `pin_memory=False` and `num_workers=0`, including on Windows.

The loop prints **measured** seconds for every update and the running mean; no
runtime estimate is assumed. The budget starts after initial validation and its
baseline checkpoint. It is checked between updates and before validation. One
in-flight update or validation can exceed the budget. Model loading, initial
validation, and saving add wall-clock time. A final validation runs only if there
is time left; otherwise the previously validated best remains selected. The
test split is deliberately not evaluated automatically.

The objective is MSE between generated-image VGG relu4_1 features and the AdaIN
target, plus the summed MSEs of style feature means and standard deviations at
relu1_1 through relu4_1. Target extraction uses `no_grad`. The generated image's
VGG pass remains differentiable during training even though VGG parameters are
frozen. Before every optimizer update, decoder gradients must be present, finite
and not all zero, and VGG must have no gradients.

### Outputs and resume

Everything produced by an experiment is under `experiment/<experiment>/`:

- `decoder_final.pth`: best validated decoder, ready for Flask. The starting
  decoder is saved at step 0 and retained if no validated update improves it.
- `latest.pth`: last completed saved step's decoder, Adam state, progress, RNGs,
  sampler positions, original configuration, split, and best decoder. It is
  separate from the best export and is saved initially, at validation, every
  `--checkpoint_every` updates (100 by default), and at normal completion,
  including budget/early stopping. An interrupted process can
  resume from its last completed checkpoint; it may lose intervening updates.
- `losses.csv`: per-update and validation content/style/weighted losses and
  measured seconds. On resume it is reconstructed from the checkpoint history.
- `previews/`: fixed validation content/style/output triplets for the initial,
  best and most recent validated decoder; output pixels are clamped only for
  display, never for the training loss.
- `config.json`, `selection.json`, `summary.json`, `selected_files.json`:
  settings, selected step/loss, measured results and selected filenames.

Exports and training checkpoints are written atomically per file. `latest.pth`
is the authoritative resume state; the best export and CSV are reconstructed
from it on resume. Decoder exports are strictly reloaded and compared at normal
completion. No earlier training outputs are silently replaced by a fresh run:
choose a new `--experiment`, or explicitly resume:

```powershell
# Reuse the original data/transform/loss/optimizer settings.
# The time budget restarts for this invocation; max_updates is a total target.
.\myenv\Scripts\python.exe train.py --experiment cpu_pilot --resume experiment/cpu_pilot/latest.pth --max_updates 150 --time_budget 300 --threads 2
```

`--resume` cannot be combined with `--decoder_path` or
`--allow_random_decoder`. Resume checks the selected data and relevant settings;
an already-triggered early-stopping condition remains stopped. For a new
fine-tuning stage, use a new experiment and pass the old best export as
`--decoder_path` instead. Flask accepts `NST_DECODER_PATH`, `NST_VGG_PATH` and
`NST_CPU_THREADS` environment variables; relative model paths resolve against
the project directory, even when launched from another working directory.

### Initial pilot verification

Syntax checks passed for `train.py`, `app.py`, `utils/models.py` and
`utils/utils.py`. The real dataset check found 128 content and 8,476 style images
and successfully decoded all 256 selected images into `(1, 3, 128, 128)` batches.
Both published pretrained assets were downloaded, inspected and strictly loaded.
A separate `_smoke_cpu` experiment ran one real 128-pixel optimizer update,
measuring 1.093 seconds for that update on this machine; this is a measurement,
not an estimate for another machine or a full run. Its final validation and
decoder export/reload passed. The main `cpu_pilot` experiment later completed
100 updates and selected step 25 as its best validation checkpoint. Its
artifacts are retained.

Temporary small fixtures verified mixed-case extensions, RGB conversion,
unreadable-file errors, split disjointness, insufficient-data rejection,
unequal collection sizes, finite/nonzero decoder gradients, no VGG gradients,
strict checkpoint rejection/reload, retention of the initial decoder, early
stopping and a near-zero time budget. Two uninterrupted updates exactly matched
one update followed by resume in decoder weights, Adam state, random state and
sampler position. Download integrity and no-overwrite behavior were exercised
using local copies of the verified release bytes and a deliberately corrupt
response. The test report is `experiment/_verification/results.json`.

The original Flask test used stubbed template rendering because the template
was initially absent. The restored template now passes a real home-page request.
CPU checkpoint refresh and backward-compatible pilot resume also pass. The
larger-profile smoke test measured 2.155 seconds for one 256-pixel CPU update;
this is a single measurement, not a runtime guarantee. See
`experiment/_verification/6000_checks.json` for those checks.

## Training with 6,000 content and 6,000 style images

The `6000` profile keeps 6,000 training images **from each collection**, plus
14 validation and 14 test images from each. That requires 6,028 photos and
6,028 paintings. The content downloader adds only the missing COCO train2017
photos, retaining existing files. It uses a reproducible filename selection,
HTTPS, JPEG decoding checks, per-file SHA-256 records, retries and atomic file
publication. Its plan and progress are in
`experiment/adain_6000/content_download.json`; rerunning retries incomplete
downloads. The 118,287-image filename list comes from the
[Ultralytics COCO label archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip);
only the list is retained. Selected images come from COCO's S3 image bucket.
The full COCO image archive is never downloaded.

This profile starts from the published pretrained decoder, uses 256-pixel
crops after resizing the shorter edge to 512, batch size 1, Adam at `1e-5`, and
CPU execution. Larger crops provide more spatial context; the
[published AdaIN trainer](https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py)
also uses 256-pixel training crops. This project's resize preserves aspect ratio.
The model architectures and loss objectives remain unchanged.

The update target is 2,000 (one-third of a pass through each training
collection at batch size 1). Validation runs every 500 updates. The
five-minute pilot limit is disabled (`--time_budget 0`) for this profile. This
is a substantial CPU run; use its measured update times to monitor progress.
More images or a lower validation loss do not guarantee better-looking faces
or fine detail. Compare the initial and best previews. At inference, reducing
the Style Strength slider blends more content features into the result; try
0.6 to 0.8 if full strength removes too much detail.

From this project directory, run:

```powershell
# Or run all three preparation/training stages in one foreground command:
# .\myenv\Scripts\python.exe -u train.py --profile 6000 --prepare_and_train

# Complete/retry the partial COCO download.
.\myenv\Scripts\python.exe train.py --profile 6000 --download_content

# Decode all selected images and verify their 256-pixel tensors.
.\myenv\Scripts\python.exe train.py --profile 6000 --check_data

# Start a fresh larger run; the original pilot remains intact.
.\myenv\Scripts\python.exe -u train.py --profile 6000

# If interrupted, continue with exactly the saved split/settings.
.\myenv\Scripts\python.exe -u train.py --profile 6000 --resume experiment/adain_6000/latest.pth
```

A background run has been queued for the requested experiment. Do not start a
second fresh run while it is active. It waits for the current download, retries
missing photos if needed, checks all selected images, then starts training.
Keep the computer awake while it runs. Monitor it with:

```powershell
Get-Content experiment/adain_6000/run_status.json
Get-Content experiment/adain_6000/training.log -Tail 10 -Wait
```

`run_status.json` distinguishes waiting, downloading, checking, training,
finished and failed stages. `training-errors.log` contains process errors, and
`background_launcher.json` records the background process and command. A failed
check stops the pipeline before training; completed downloads are retained.

All profile defaults can be overridden explicitly. `--time_budget 3600` limits
one invocation's loop to about an hour; repeat the resume command to continue.
To request a graceful stop between updates, create
`experiment/adain_6000/stop_requested`; remove that file before resuming.
An abrupt interruption can lose updates after the last completed checkpoint.

Only a completed run that has processed at least one pass writes
`experiment/adain_6000/ready_for_app.json`. With no `NST_DECODER_PATH` override,
Flask keeps using the pilot until that marker appears, then strictly loads the
larger experiment's `decoder_final.pth` on the next request. The export remains
the best validated decoder, possibly the starting decoder if no update improves
the selection loss. Clear an old override when launching to enable this:

```powershell
Remove-Item Env:NST_DECODER_PATH -ErrorAction SilentlyContinue
.\myenv\Scripts\python.exe app.py
```

An explicit override always wins. For example, after the larger run completes:

```powershell
$env:NST_DECODER_PATH = 'experiment/adain_6000/decoder_final.pth'
.\myenv\Scripts\python.exe app.py
```
