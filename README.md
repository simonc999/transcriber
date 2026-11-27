# Meeting Transcriber

Turn a saved English meeting recording into a timestamped transcript, subtitles, and word-level review scores. The command works with audio and video files such as MP3, WAV, M4A, and MP4. It runs locally on a CPU by default and can add speaker labels if you install the optional speaker model.

## Where to run it

| Folder | Use it for |
| --- | --- |
| [`local_cpu/`](local_cpu/README.md) | Transcription on your WSL machine using its CPU. |
| [`local_gpu/`](local_gpu/README.md) | Transcription on your WSL machine using an NVIDIA GPU. |
| [`drive/`](drive/README.md) | Transcription on a Google Colab GPU, with recordings and transcripts in Google Drive. |

The `main` branch contains all three workflows and their shared code under `src/meeting_transcriber/`. The local commands can read recordings from any path. In Colab, the notebook clones the code into the runtime and reads and saves files through the mounted Drive folder.

## Clone one workflow

Each workflow branch contains the shared Python package and the files for one environment.

| Branch | Workflow |
| --- | --- |
| `local-cpu` | CPU on WSL |
| `local-gpu` | NVIDIA GPU on WSL |
| `colab-drive` | Colab GPU with Google Drive |

From a Git remote that contains the branches, choose one with a shallow clone. Replace `YOUR_REPO_URL` with the Git clone URL.

```bash
git clone --depth 1 --single-branch --branch local-cpu YOUR_REPO_URL
```

The downloadable ZIP includes the branches. To clone directly from an extracted ZIP, use the path to that checkout as the source and add `--no-local`.

```bash
git clone --no-local --depth 1 --single-branch --branch local-cpu \
  "$HOME/meeting-transcriber" meeting-transcriber-cpu
```

To make them available from your Git host, configure `origin` and push the branches from the full checkout.

```bash
git push -u origin main local-cpu local-gpu colab-drive
```

The Colab notebook selects `colab-drive` from either the ZIP or the configured Git URL.

## Install on WSL

Run these commands in your **Ubuntu terminal in WSL**. Ubuntu 24.04 with Python 3.12 is a good starting point. The project requires Python 3.10 or newer.

Download `meeting-transcriber.zip`. In the commands below, replace `YourName` with the name of your Windows user folder, or substitute the path to the ZIP.

```bash
sudo apt update
sudo apt install -y python3 python3-venv unzip

cd ~
unzip "/mnt/c/Users/YourName/Downloads/meeting-transcriber.zip"
cd meeting-transcriber

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
meeting-transcriber --help
```

The ZIP contains the working tree and Git history. To inspect the history, install Git with `sudo apt install -y git` and run `git log --oneline` from the project directory.

## Transcribe a recording

Replace the example path with your recording. Keep quotes around paths that include spaces. A Windows file on the C drive is available in WSL under `/mnt/c/`.

```bash
meeting-transcriber "/mnt/c/Users/YourName/Downloads/weekly-sync.mp4"
```

The first run downloads the Whisper model. The model is cached for later runs, and your recording is processed on your machine. A new terminal needs `cd ~/meeting-transcriber` followed by `source .venv/bin/activate` before you run the command again.

Once Whisper loads the model and prepares the audio, the progress bar starts at 0%. It advances as Whisper returns text and can jump over silence. Terminals that support redrawing show one updating line. Basic terminals and redirected logs show occasional progress lines. Use `--no-progress` to hide the bar.

For `weekly-sync.mp4`, open the files under `transcripts/weekly-sync/`.

| File | Use |
| --- | --- |
| `transcript.txt` | Read the timestamped transcript and find passages marked `[review]`. |
| `transcript.srt` | Load subtitles alongside the original recording. |
| `transcript.json` | Inspect utterance times, review flags, and individual word scores and times. Times are in seconds. |

`[review]` indicates a word below the review threshold or a passage with incomplete word alignment. The default threshold is `0.5`. Raise it with `--review-threshold 0.65` to flag more passages. The scores are model estimates. Listen back to verify names, dates, and decisions, especially in noisy or overlapping speech.

## Choose a model

List the built-in model names without downloading them.

```bash
meeting-transcriber --list-models
```

| Model | When to use it |
| --- | --- |
| `tiny.en` | Quick English draft. |
| `small.en` | Default for everyday English meetings on a CPU. |
| `medium.en` | Difficult English recordings when you have more processing time and memory. |
| `large-v3` | Larger multilingual model, typically run on a compatible GPU. |

Select a model with `--model`. You can also supply names, acronyms, and technical vocabulary with `--hotwords`.

```bash
meeting-transcriber "/mnt/c/Users/YourName/Downloads/weekly-sync.mp4" \
  --model medium.en --hotwords "Kubernetes, Acme, Priya" \
  --output-dir transcripts/medium
```

Give each retry or recording with the same filename its own `--output-dir` to keep earlier results. Follow the [local NVIDIA GPU setup](local_gpu/README.md) or the [Colab workflow](drive/README.md) to run a larger model on a GPU.

## Add speaker labels

1. Sign in to Hugging Face, accept the model conditions, and create a read token from the same account. [Speaker model](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Access tokens](https://huggingface.co/settings/tokens)
2. In the project's active virtual environment, install the matching CPU packages and speaker dependencies.

   ```bash
   sudo apt install -y ffmpeg
   python -m pip install torch==2.8.0 torchaudio==2.8.0 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -e '.[speakers]'
   ```

3. Enter the token at the hidden prompt and run the command. Use `--speakers 3` if you know exactly three people speak. Omit it to estimate the count.

   ```bash
   read -rsp "Hugging Face read token: " HF_TOKEN
   echo
   export HF_TOKEN
   meeting-transcriber "/mnt/c/Users/YourName/Downloads/weekly-sync.mp4" \
     --diarize --speakers 3 --output-dir transcripts/speakers
   unset HF_TOKEN
   ```

The labels are anonymous and specific to each recording, such as `Speaker 1`. A passage without a matching detected turn is labeled `Unknown speaker`. Check the labels where people interrupt or speak at the same time. The speaker model runs on the CPU.

## Troubleshooting

| What you see | What to check |
| --- | --- |
| `meeting-transcriber: command not found` | Activate `.venv` in the project directory and check that `python -m pip install -e .` finished. |
| The progress bar is missing | Check `python -m pip show meeting-transcriber` for version 0.2.1 or newer. The first model download can take time before the bar starts. |
| `Recording file not found` | Check the Linux path, Windows user folder, filename, and extension. |
| `Could not decode audio` | Open the recording in a player and verify it has an audio track. |
| The speaker model returns HTTP 401 or 403 | Accept the model conditions and enter a token from that same Hugging Face account. |
| A CUDA library fails to load | Check the [GPU setup](local_gpu/README.md) and launch with `bash local_gpu/run.sh`. |

Run `meeting-transcriber --help` to see all options.
