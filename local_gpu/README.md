# Local NVIDIA GPU on WSL

Use WSL 2 and an NVIDIA GPU with a current **Windows** NVIDIA driver. Run `nvidia-smi` in your WSL terminal and check that it lists your GPU. If the command is missing from your PATH, also try `/usr/lib/wsl/lib/nvidia-smi`. [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Follow the [WSL installation](../README.md#install-on-wsl) through creation of the virtual environment. Run these commands from the repository root with that environment active. The GPU extra installs the CUDA 12 cuBLAS and cuDNN 9 libraries. The launcher makes those libraries available to Whisper and requests CUDA. [Faster Whisper GPU setup](https://github.com/SYSTRAN/faster-whisper#gpu)

```bash
source .venv/bin/activate
python -m pip install -e '.[gpu]'
bash local_gpu/run.sh "/mnt/c/Users/YourName/Downloads/weekly-sync.mp4" \
  --model large-v3 --output-dir transcripts/local_gpu
```

Results are saved under `transcripts/local_gpu/weekly-sync/`. Try `--model medium.en` or `--model small.en` if the larger model exceeds your GPU memory. The launcher accepts the same options as `meeting-transcriber`, including `--hotwords` and `--diarize`. Install the [optional speaker dependencies](../README.md#add-speaker-labels) before using `--diarize`.
