# Contributing to RayLLM-Orchestrator

Thanks for your interest in contributing! Here's how to help.

## What We Need

### Bugs & Fixes (Most Valuable)
- Found a crash or wrong output? File an issue with reproducible steps.
- Have a fix? Open a PR with the test case included.

### New Hardware Support
- Works great on your GPU/Mac but we don't have it documented?
- Add instructions to README.md
- Include: hardware specs, expected tok/s, any special config

### Documentation
- Unclear part of the README? Suggest clearer wording.
- Found outdated examples? Update them.
- Wrote a tutorial or blog post? Link it in an issue.

### Performance Improvements
- Faster inference on your hardware? PR with benchmark numbers.
- Better memory usage? Show before/after on a dataset.

### What We Don't Need
- Major architectural rewrites without discussion
- New CLI commands or flags without an issue first
- Features that break simplicity (we delete bloat, not add features)
- Cosmetic changes that don't improve usability

## Development Setup

```bash
# Clone
git clone https://github.com/agastya-choudhary123/rayllm-orchestrator
cd rayllm-orchestrator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Test
python orchestrator.py train --help
python orchestrator.py serve --help
```

## How to Submit

1. **Find or create an issue** — Describe what you want to fix/add
2. **Fork the repo** — Click "Fork" on GitHub
3. **Create a branch** — `git checkout -b fix/your-fix-name`
4. **Make changes** — Keep commits focused and clear
5. **Test locally** — Run the code, make sure it works
6. **Push & open PR** — Reference the issue in your PR description
7. **Respond to feedback** — We'll review and suggest improvements

## Code Style

- No complex abstractions — prefer clear code over clever code
- No comments unless the WHY is non-obvious
- Functions should do one thing
- Keep it simple enough for someone new to understand

## Testing

Before pushing, test the core commands:

```bash
# Train
python orchestrator.py train --model gpt2 \
  --dataset examples/my-data.jsonl --epochs 1

# Serve
python orchestrator.py serve --model ./checkpoints/gpt2 --port 8000

# In another terminal
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"test"}]}'
```

## Questions?

Open an issue and ask. We don't bite.

## License

By contributing, you agree your code is licensed under MIT.
