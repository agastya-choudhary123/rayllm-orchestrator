# How to Ship RayLLM-Orchestrator

**Goal:** Get this tool to users via PyPI (pip install) and Docker (docker pull).

**Timeline:** 1-2 weeks to v0.1.0

---

## Phase 1: Prepare (This Week)

### 1. Create GitHub Repository

```bash
# Initialize git (if not already done)
cd rayllm-orchestrator
git init
git add .
git commit -m "Initial commit: clean LLM training and serving tool"

# Create repo on GitHub
# Visit https://github.com/new
# Name: rayllm-orchestrator
# Description: Train and serve LLMs on any hardware
# License: MIT
# Make it public

# Push to GitHub
git remote add origin https://github.com/yourorg/rayllm-orchestrator.git
git branch -M main
git push -u origin main
```

### 2. Set Up pyproject.toml

Create `pyproject.toml` at the repo root:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rayllm-orchestrator"
version = "0.1.0"
description = "Train and serve LLMs on any hardware with one command each"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
requires-python = ">=3.9"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "datasets>=2.20",
    "peft>=0.12",
    "accelerate>=0.33",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]
gpu = [
    "ray[default]>=2.35",
    "vllm>=0.6.0",
    "bitsandbytes>=0.43",
]
mlx = ["mlx>=0.18"]

[project.scripts]
rayllm = "orchestrator:main"

[project.urls]
Homepage = "https://github.com/yourorg/rayllm-orchestrator"
Repository = "https://github.com/yourorg/rayllm-orchestrator"
```

### 3. Reorganize Code (Optional)

If you want `pip install rayllm-orchestrator` to work cleanly:

```bash
# Create src/ structure (optional but cleaner)
mkdir -p src/rayllm
mv orchestrator/* src/rayllm/
mv orchestrator.py src/rayllm/__main__.py

# Update imports (optional)
```

For now, just ensure:
- `orchestrator.py` has `if __name__ == "__main__": sys.exit(main())`
- `orchestrator/` has `__init__.py` (already exists)

### 4. Add .gitignore (if not present)

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data
checkpoints/
*.jsonl
*.pkl

# OS
.DS_Store
Thumbs.db
EOF
```

### 5. Test Locally

```bash
# Install in editable mode
pip install -e .

# Test the CLI
rayllm --help
rayllm train --help
rayllm serve --help

# Test imports
python -c "from orchestrator import train, serve; print('✓ Works')"
```

---

## Phase 2: Publish to PyPI (Next Week)

### 1. Create PyPI Account

- Go to https://pypi.org
- Sign up (or log in)
- Create an API token: Settings → API Tokens → Create
- Copy the token (you'll use it once)

### 2. Build the Package

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Check the build
ls -lh dist/
# Should show:
#   rayllm-orchestrator-0.1.0.tar.gz (source)
#   rayllm-orchestrator-0.1.0-py3-none-any.whl (wheel)
```

### 3. Upload to PyPI (First Time)

```bash
# Upload (enter token as password when prompted)
twine upload dist/*

# Username: __token__
# Password: pypi-AgEIcHlwaS5vcmc...
```

**Done!** Users can now install:
```bash
pip install rayllm-orchestrator
rayllm train --model gpt2 --dataset my-data.jsonl
```

### 4. Create GitHub Release

```bash
# Tag the release
git tag v0.1.0
git push origin v0.1.0

# Create release on GitHub
# Visit: https://github.com/yourorg/rayllm-orchestrator/releases/new
# Tag: v0.1.0
# Title: v0.1.0
# Description:
# - Initial release
# - Train: fine-tune any model on your data
# - Serve: auto-selects MLX/vLLM/transformers backend
# - Works on Mac/GPU/CPU

# Publish
```

---

## Phase 3: Docker (Optional, Phase 2)

### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for PyTorch + torch extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY . .
RUN pip install --no-cache-dir -e .

# Default: train (override with: docker run <image> serve ...)
ENTRYPOINT ["rayllm"]
CMD ["train", "--help"]
```

### 2. Build & Test Locally

```bash
docker build -t rayllm:test .
docker run -it rayllm:test train --help
docker run -it rayllm:test serve --help
```

### 3. Push to Docker Registry

```bash
# Option A: Docker Hub
docker tag rayllm:test yourorg/rayllm:0.1.0
docker push yourorg/rayllm:0.1.0

# Option B: GitHub Container Registry (ghcr.io)
docker tag rayllm:test ghcr.io/yourorg/rayllm:0.1.0
docker push ghcr.io/yourorg/rayllm:0.1.0
```

Users can now run:
```bash
docker run -it ghcr.io/yourorg/rayllm:0.1.0 train \
  --model gpt2 --dataset my-data.jsonl
```

---

## Phase 4: Automate (GitHub Actions)

### 1. Auto-Publish to PyPI on Release

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Build
        run: |
          pip install build twine
          python -m build
      
      - name: Publish
        run: twine upload dist/* -u __token__ -p ${{ secrets.PYPI_TOKEN }}
```

### 2. Auto-Build Docker Image

Create `.github/workflows/docker.yml`:

```yaml
name: Build Docker Image

on:
  push:
    tags:
      - "v*"

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v2
      
      - name: Login to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build & Push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### 3. Set Up PyPI Token

```bash
# Go to: https://github.com/yourorg/rayllm-orchestrator/settings/secrets/actions
# New Repository Secret:
# Name: PYPI_TOKEN
# Value: (paste your token from https://pypi.org/account/api-tokens/)
```

### 4. From Now On

To release v0.2.0:
```bash
# Update version in pyproject.toml
# Commit
git add pyproject.toml
git commit -m "Bump to v0.2.0"

# Tag (triggers auto-publish)
git tag v0.2.0
git push origin main
git push origin v0.2.0

# GitHub Actions automatically:
# 1. Publishes to PyPI
# 2. Builds & pushes Docker image to ghcr.io
```

---

## Checklist for v0.1.0

- [ ] `pyproject.toml` created with correct metadata
- [ ] `pip install -e .` works
- [ ] `rayllm train --help` works
- [ ] `rayllm serve --help` works
- [ ] README.md is clear and complete
- [ ] `.gitignore` covers Python artifacts
- [ ] GitHub repo created and pushed
- [ ] PyPI account + token created
- [ ] Package built locally: `python -m build`
- [ ] Package uploaded: `twine upload dist/*`
- [ ] GitHub release created (v0.1.0)
- [ ] Tag pushed: `git push origin v0.1.0`

---

## Testing After Release

```bash
# Test PyPI install
pip install rayllm-orchestrator
rayllm train --help

# Test Docker
docker run -it ghcr.io/yourorg/rayllm:0.1.0 serve --help
```

---

## Troubleshooting

### "twine: command not found"
```bash
pip install twine
```

### "Failed to upload: Invalid distribution"
```bash
# Check the build
twine check dist/*

# Rebuild
rm -rf dist/ build/
python -m build
```

### Docker push fails
```bash
# Check login
docker login ghcr.io
# Username: yourorg
# Password: (GitHub personal access token with write:packages)
```

### GitHub Actions workflow fails
- Check logs: https://github.com/yourorg/rayllm/actions
- Common issue: `PYPI_TOKEN` secret not set
- Check: Settings → Secrets → Actions → PYPI_TOKEN exists

---

## After v0.1.0

1. **Listen to users** — GitHub Issues for bugs/requests
2. **Fix bugs** — Release v0.1.1, v0.1.2 as needed
3. **Add features** — Based on user feedback, not guesses
4. **v1.0.0** — When the tool is stable and proven

Don't over-engineer. Release, iterate, ship.
