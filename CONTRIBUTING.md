# 🤝 Contributing to Nina IA

Thank you for your interest in contributing to **Nina IA**! This guide explains how to set up your development environment, follow the project conventions, and submit your contributions.

---

## 📋 Table of Contents

- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Commit Conventions](#-commit-conventions)
- [Running Tests](#-running-tests)
- [Submitting a Pull Request](#-submitting-a-pull-request)
- [Reporting Bugs](#-reporting-bugs)

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Nina-IA.git
   cd Nina-IA
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/Marcos-Felipe-dos-Santos/Nina-IA.git
   ```
4. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feat/my-new-feature
   ```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- FFmpeg (added to PATH)
- espeak-ng (required by Kokoro TTS)
- NVIDIA GPU (recommended for WhisperX)

### Environment

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env     # Windows
cp .env.example .env       # Linux/macOS
```

Edit the `.env` file with your API keys (see `.env.example` for reference).

> **Windows users:** run `install.bat` for guided dependency installation with CUDA support.

---

## 📝 Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/). Every commit message should follow this format:

```
<type>(scope): <description>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `perf` | Performance improvements |
| `refactor` | Code refactoring (no feature or fix) |
| `test` | Adding or updating tests |
| `chore` | Build, CI, or tooling changes |
| `style` | Code style changes (formatting, semicolons, etc.) |

### Examples

```
feat(stt): add microphone device selection via config
fix(tts): resolve audio clipping on long sentences
docs: update README with VTube Studio setup guide
perf(pipeline): wrap TTS in asyncio.to_thread to avoid blocking
refactor(memory): implement singleton pattern for MemoryManager
test(avatar): add emotion detection performance benchmarks
chore: update dependencies in requirements.txt
```

---

## 🧪 Running Tests

Always run the test suite before opening a Pull Request:

```bash
# Run all pytest-compatible tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_avatar.py -v          # Avatar & emotion detection
python -m pytest tests/test_optimizations.py -v    # Performance & architecture

# Run standalone test scripts
python tests/test_fase3.py    # ChromaDB memory (requires sentence-transformers)
python tests/test_fase4.py    # Tool calling (requires duckduckgo-search)
```

> **Note:** Some tests require hardware (GPU for WhisperX, microphone for STT). Tests that need external services will skip gracefully if the dependency is unavailable.

---

## 🔀 Submitting a Pull Request

1. **Ensure tests pass** on your branch.
2. **Update documentation** if your change affects usage or setup.
3. **Keep PRs focused**: one feature or fix per PR.
4. **Write a clear PR description** explaining what and why.
5. **Reference issues**: Use `Fixes #123` or `Closes #456` if applicable.

### PR Checklist

- [ ] Tests pass locally (`python -m pytest tests/ -v`)
- [ ] Code follows project style (clean, readable, no unnecessary comments)
- [ ] Commit messages follow conventional commits
- [ ] Documentation updated if needed
- [ ] No hardcoded secrets or API keys

---

## 🐛 Reporting Bugs

Found a bug? Open an [issue](https://github.com/Marcos-Felipe-dos-Santos/Nina-IA/issues/new) using this template:

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run '...'
2. Say '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots / Logs**
If applicable, add screenshots or paste the terminal output.

**Environment:**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.9]
- GPU: [e.g., RTX 4060 Ti]
- LLM Provider: [e.g., Gemini / Ollama]

**Additional context**
Add any other context about the problem here.
```

---

## 💜 Thank You!

Every contribution — whether it's code, docs, bug reports, or ideas — makes Nina IA better. Thank you for being part of this project!
