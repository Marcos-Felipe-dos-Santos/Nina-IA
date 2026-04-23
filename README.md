<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LLM-Gemini_&_Ollama-4285F4?logo=google&logoColor=white" alt="LLMs">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-active-success" alt="Status">
  <img src="https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white" alt="Windows">
  <br>
  <a href="https://github.com/Marcos-Felipe-dos-Santos/Nina-IA/actions/workflows/tests.yml">
    <img src="https://github.com/Marcos-Felipe-dos-Santos/Nina-IA/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
</p>

<p align="center">
  <!-- TODO: Grave uma demonstração (Screen Recording) e transforme em GIF. Salve em docs/nina-demo.gif e descomente a linha abaixo -->
  <!-- <img src="docs/nina-demo.gif" alt="Nina IA Demo" width="600"> -->
  <br>
  <em>🤖 Nina IA — Your Local-First Autonomous AI Voice Assistant</em>
</p>

---

## 🌍 English

**Nina IA** is a modular, mostly local-first AI voice agent built in Python. She listens to your voice, thinks with Google Gemini or local Llama 3 (via Ollama), remembers past conversations permanently with semantic memory (RAG), executes real-world actions on your PC, sees your screen, and speaks back to you — all monitored through a beautiful real-time Web Dashboard.

## 🇧🇷 Português

**Nina IA** é uma agente pessoal de voz modular e predominantemente *local-first*, construída em Python. Ela ouve sua voz, pensa com o Google Gemini ou Llama 3 local (via Ollama), lembra de conversas anteriores permanentemente com memória semântica (RAG), executa ações reais no PC, enxerga sua tela e responde por voz — tudo isso com um painel (Dashboard) em tempo real.

---

### ✨ Features & Status

| Feature | Technology | Status |
|---------|-----------|--------|
| 🎤 Voice Input (STT) | WhisperX + silero-vad | ✅ |
| 🔊 Voice Output (TTS) | Kokoro TTS (pt-BR) | ✅ |
| 🧠 AI Reasoning | Google Gemini 2.0 Flash / **Ollama** (Llama 3) | ✅ |
| 📚 Long-term Memory | ChromaDB + sentence-transformers (RAG) | ✅ |
| 🔧 Tool Calling | Native function calling for AI Agent actions | ✅ |
| 💻 System Mastery | Opens apps, web searches, checks time/date | ✅ |
| 📝 Note Taking | Create & list local notes | ✅ |
| 👁️ Screen Vision | mss capture + Gemini Vision (multimodal) | ✅ |
| 🌐 Real-time Dashboard | FastAPI + WebSocket + Vanilla JS + CSS | ✅ |

---

## 🏗️ Architecture Stack

```text
┌────────────────────────────────────────────────────────────┐
│                      Nina IA Pipeline                       │
│                                                            │
│  🎤 Microfone ──▶ VAD (silero) ──▶ STT (WhisperX)         │
│        │                                                   │
│        ▼                                                   │
│  📚 Busca Semântica (ChromaDB + sentence-transformers)    │
│        │                                                   │
│        ▼                                                   │
│  🧠 LLM Inteligência (Gemini Flash ou Ollama Llama 3)     │
│        │                                                   │
│        ├──▶ 🔧 Tools (ações locais do sistema operac.)     │
│        │                                                   │
│        ├──▶ 👁️ Visão (Screenshot mss + Gemini Vision)      │
│        │                                                   │
│        ▼                                                   │
│  💾 Salvar Memória (ChromaDB)                             │
│        │                                                   │
│        ▼                                                   │
│  🔊 Sintetização de Voz (Kokoro TTS pt-BR)                 │
│                                                            │
│  📡 EventBus ──────▶ 🌐 Dashboard (FastAPI / WebSocket)   │
└────────────────────────────────────────────────────────────┘
```

> **Tech Highlights:**
> - **Gemini API** for cloud reasoning & multimodal vision.
> - **Llama 3 via Ollama** for offline, local fallback reasoning.
> - **WhisperX** for lightning-fast speech recognition.
> - **Kokoro TTS** for natural Brazilian Portuguese voice.
> - **ChromaDB + sentence-transformers** for vectorized memory retrieval.
> - **FastAPI** for real-time WebSocket dashboarding.

---

## 💻 Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | — | **NVIDIA RTX 3060 / 4060 Ti** (CUDA) |
| **Storage** | 5 GB free | 20 GB free (SSD) |

> **Note**: An NVIDIA GPU (RTX series recommended) is heavily suggested to achieve near-instant STT conversions using WhisperX and to run local Ollama models with no latency.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg** — Add to PATH ([download](https://ffmpeg.org/download.html))
- **espeak-ng** — Required by Kokoro TTS ([download](https://github.com/espeak-ng/espeak-ng/releases))
- *(Optional)* **Ollama** — Download at [ollama.com](https://ollama.com/download), install, and run `ollama pull llama3`.

### 2. Environment Setup
```bash
git clone https://github.com/Marcos-Felipe-dos-Santos/Nina-IA.git
cd Nina-IA

# Crie e ative a virtual environment
python -m venv venv
.\venv\Scripts\activate    # No Windows

# Instale os requerimentos do projeto
pip install -r requirements.txt
```

> **Windows users:** run `install.bat` for guided dependency installation with CUDA support.

### 3. Environment Variables (`.env`)
Configurações sensíveis e provedor de IA são lidos de um arquivo oculto chamdo `.env`. O projeto ignora este arquivo via `.gitignore` para sua segurança.

Dê um copy no template `env.example`:
```bash
copy .env.example .env
```
Abra o `.env` gerado e defina suas chaves:
```env
GEMINI_API_KEY=AIzaSy...sua_chave_aqui
LLM_PROVIDER=gemini # ou altere para "ollama" para modo Offline
```

### 4. Configuration (`config.yaml`)
Para alterar as propriedades do sistema e comportamentos da Nina, use o `config.yaml`. (O main usa esse arquivo para orquestrar dependências, ex: `small` no STT, `llama3` no Ollama, compressão do Print de Tela, etc).

### 5. Running
```bash
python main.py
```
> O terminal carregará a orquestração e mostrará o Dashboard Vivo disponível no seu navegador em: **`http://localhost:8000`**.

---

## 🎭 Avatar VTube Studio

A Nina possui integração visual direta com o **VTube Studio** via WebSocket (pyvts), reagindo em tempo real e mudando expressões faciais com base no que ela mesma falou.

### Como configurar:
1. Abra o VTube Studio e vá em **Settings** → **Plugins** → **Enable Plugin API**.
2. No VTube Studio, crie as "Hotkeys" com os mesmos nomes definidos no `config.yaml` (Ex: "Sorrir", "Triste", "Surpresa").
3. Habilite na configuração editando o arquivo `config.yaml`:
   ```yaml
   avatar:
     enabled: true
   ```
4. Ao executar a `main.py` pela primeira vez, aprove o popup de permissão que aparecerá dentro da interface do VTube Studio.

<!-- Placeholder para screenshot/GIF do avatar em ação -->
<!-- <img src="docs/avatar-demo.gif" alt="Nina IA Avatar Demo" width="600"> -->

---

## 🗺️ Roadmap

- [x] Pipeline de voz completo (STT → LLM → TTS)
- [x] Memória de longo prazo com RAG (ChromaDB)
- [x] Agente com 6 tools (function calling)
- [x] Visão de tela multimodal (Gemini Vision)
- [x] Dashboard web em tempo real (FastAPI + WebSocket)
- [x] Modo offline 100% local (Ollama + Llama 3)
- [x] Integração com VTube Studio (avatar com emoções)
- [ ] PNGTuber support (avatar sem Live2D)
- [ ] Discord bot integration
- [ ] RVC voice covers (play_audio tool)
- [ ] Multi-language support (EN/ES)

---

## 💡 Inspired By
- **Neuro-Sama** by Vedal — the incredible VTuber pioneer entity.
- **[kimjammer/Neuro](https://github.com/kimjammer/Neuro)** — amazing reference project for autonomous virtual agents and pipeline routing!

---

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md) for details on how to fork, set up the development environment, and submit pull requests.

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. You are allowed to copy, modify, and distribute this system unconditionally. Made with 💜 in Brazil.
