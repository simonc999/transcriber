# Local CPU on WSL

Follow the [WSL installation](../README.md#install-on-wsl) once. Run the command from the repository root with the virtual environment active. Use a path to your recording.

```bash
source .venv/bin/activate
meeting-transcriber "/mnt/c/Users/YourName/Downloads/weekly-sync.mp4" \
  --model small.en --output-dir transcripts/local_cpu
```

Results are saved under `transcripts/local_cpu/weekly-sync/`. Try `--model medium.en` for a difficult recording when you can spare more processing time. Use `meeting-transcriber --list-models` to list the model names.
