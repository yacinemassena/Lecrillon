# How to Build and Upload Custom Mamba Wheels to R2

Since compiling `mamba-ssm` and `causal-conv1d` requires a specific CUDA environment matching your target hardware (RTX 5080 / sm_120), you don't need to create a *new* special venv just for building. You can use the WSL environment that you have already set up using your existing `setupenv.sh` script, which has all the correct CUDA paths and PyTorch versions installed.

Follow these steps to build the wheels and upload them to your Cloudflare R2 bucket.

## Step 1: Open WSL and Activate the Environment

Open your WSL terminal and navigate to your project directory.

```bash
cd "/mnt/d/Mamba v2"
```

Activate the virtual environment that was created by `setupenv.sh`.

```bash
source venv/bin/activate
```

*(If you haven't run `setupenv.sh` yet on your WSL machine, you'll need to run that first to create the `venv` and install the base dependencies).*

## Step 2: Build the Wheels

Run the build script to compile the packages into `.whl` files. This step might take a few minutes as it compiles the CUDA kernels.

```bash
bash build_wheels.sh
```

You should see the compiled `.whl` files in the `custom_wheels/` directory once this finishes.

## Step 3: Upload to Cloudflare R2

Next, upload the built wheels to your R2 bucket (`europe/custom_mamba_wheels/`) using the Python upload script.

Ensure you have `boto3` installed (it should already be installed from `setupenv.sh`, but just in case):

```bash
pip install boto3
```

Run the upload script:

```bash
python upload_to_r2.py
```

## Step 4: Use on the VPS

Now, when you run `setupvps.sh` on your Linux VPS, it will automatically download these pre-built wheels from your R2 bucket instead of recompiling them from source, saving you a significant amount of time!
