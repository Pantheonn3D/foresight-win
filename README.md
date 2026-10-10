# Foresight for Windows

An offline meeting notetaker for Windows, modelled on Google's Mac-only AI Edge Foresight. It records your microphone and everything your PC plays (Teams, Zoom, Meet in a browser), transcribes both on your GPU, expands your shorthand notes into full meeting notes, and answers questions across past meetings and folders you link.

It uses the same open-weight model families Google named for the Mac app: **Gemma 4** for transcription, notes and answers, and **EmbeddingGemma 2** for search. Both run locally through [llama.cpp](https://github.com/ggml-org/llama.cpp). This is an independent project with no affiliation to Google.

## Install

1. Download `Foresight-Setup-x.y.z.exe` from [Releases](../../releases), or from the latest green [CI run](../../actions/workflows/ci.yml) (artifact `Foresight-Setup`).
2. Run it. It installs per-user into `%LOCALAPPDATA%\Programs\Foresight` and needs no admin rights.
   The installer isn't code-signed, so SmartScreen shows "Windows protected your PC". Click **More info → Run anyway**.
3. On first launch, Foresight downloads:
   - the llama.cpp CUDA build, from GitHub
   - Gemma 4 E4B Q8_0 plus its audio encoder, roughly 8–9 GB, from Hugging Face
   - EmbeddingGemma 2

   The setup screen shows the exact sizes before anything downloads. After that it needs no internet connection.

Requirements: Windows 10/11 x64 and the WebView2 runtime (preinstalled on Windows 11). An NVIDIA GPU with 10 GB+ VRAM is recommended for E4B. Smaller GPUs can pick Gemma 4 E2B, and non-NVIDIA GPUs use the Vulkan build.

## How it works

```
mic ──WASAPI──┐                         ┌─ "You:" lines
              ├─ 16 kHz mono ─ VAD ─ ≤28 s segments ─ Gemma 4 (audio in) ─┤
speakers ─loopback─┘                    └─ "Others:" lines ─ echo filter
your notes + transcript ─ Gemma 4 ─ enhanced notes (Markdown)
folders + meetings ─ EmbeddingGemma 2 ─ vectors in SQLite ─ top-k ─ Gemma 4 ─ answer with [n] citations
```

- **Two channels instead of diarization.** Your mic is labelled with your name and system audio is labelled "Others". That labelling is reliable for 1:1 calls and remote meetings. It can't separate several people in the same room.
- **Echo filter.** On speakers instead of headphones, your mic also hears the other side. Mic lines that closely match a system-audio line within a few seconds are dropped. Turn this off in Settings if you use headphones.
- **Segments.** Gemma 4's audio window is 30 s, so an energy VAD with a minimum-statistics noise floor cuts speech at pauses. Long monologues are cut at the quietest moment before 28 s.
- **Language.** Gemma 4's speech recognition prompt names the language, so you pick it in Settings (English, Danish and 13 others).
- **Long meetings.** If a transcript won't fit one context slot (about 20k tokens, a bit over an hour), it's condensed in chunks before notes are written.

Everything lives in `%LOCALAPPDATA%\Foresight`: models, the llama.cpp build, `foresight.db` (meetings and the search index), per-meeting FLAC audio, and logs. Delete that folder to remove all data. The uninstaller deliberately leaves it in place.

## Privacy

After setup, Foresight makes no network requests. Its two llama.cpp servers listen only on `127.0.0.1`, and the UI is served from `127.0.0.1:47821`. WebView2 is Microsoft's component and has its own update service, so verify it yourself if you need certainty, for example with Windows Firewall rules or Wireshark. Recording other people may require their consent where you live, so tell people before you record.

## Known limitations

- Audio capture uses WASAPI (`PyAudioWPatch`). Exclusive-mode apps and some Bluetooth headsets in hands-free mode can block capture. Pick a specific device in Settings if the default doesn't work.
- Gemma 4 transcription accuracy hasn't been measured against Whisper. CI checks it only against synthesized English speech.
- There is no live translation or speaker naming beyond You/Others.

## Development

```powershell
uv sync
uv run pytest -q                       # unit + API tests, no models needed
uv run python -m foresight --browser   # run from source in your browser
uv run python scripts/demo_server.py   # UI with seeded meetings and fake models (also --setup, --recording)
uv run python -m foresight selftest --wav speech.wav --expect "words you said"   # real models, end to end
uv run pyinstaller packaging/foresight.spec --noconfirm
iscc packaging\installer.iss
```

The CI `selftest` job runs on a GPU-less Windows runner. It downloads the CPU llama.cpp build and Gemma 4 E2B, synthesizes speech with Windows' built-in TTS, transcribes it, writes notes, and checks embedding search ranking.

Model weights come from their Hugging Face repositories and are covered by their own licenses; see each model card. The bundled fonts (IBM Plex Sans/Mono, Bricolage Grotesque) are under the SIL Open Font License.
