# Google Drive and Colab GPU

Google Drive stores the source recordings and finished transcripts. Colab runs the transcription on its NVIDIA GPU. The notebook clones the `colab-drive` branch into the runtime and copies the selected recording there while it runs. [Google Colab file access](https://colab.research.google.com/notebooks/io.ipynb)

## Run the notebook

1. In **My Drive**, create `meeting-transcriber/recordings/` and upload a recording there.
2. With the downloadable repo, upload the current `meeting-transcriber.zip` to `My Drive/meeting-transcriber/`. If you host the repo and its workflow branches on Git, supply the clone URL as `REPO_URL` in the notebook and skip the ZIP upload.
3. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select [`meeting_transcriber_colab.ipynb`](meeting_transcriber_colab.ipynb) from the extracted `drive/` folder.
4. In Colab, choose a GPU under **Runtime → Change runtime type**. Run the setup cell and approve the Google Drive connection. GPU availability depends on your Colab session. [Colab FAQ](https://research.google.com/colaboratory/faq.html)
5. Set `RECORDING` to the path from your project folder, such as `recordings/weekly-sync.mp4`. Run the transcription cell. It displays progress and writes the TXT, SRT, and JSON to `My Drive/meeting-transcriber/transcripts/weekly-sync/`.

The default `MODEL` in Colab is `large-v3`. Use `medium.en` or `small.en` if the model exceeds the GPU memory. Change `OUTPUT_FOLDER` to a new subfolder such as `transcripts/second-pass` to transcribe the same recording again.

## Add speaker labels

Accept the [Hugging Face speaker model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1), create a [read token](https://huggingface.co/settings/tokens), and add it to Colab **Secrets** as `HF_TOKEN`. Enable notebook access for that secret. Set `ENABLE_SPEAKERS = True` and run setup again. Use `SPEAKERS` in the transcription cell if you know the exact number of speakers. Whisper runs on the GPU and the speaker model runs on the CPU.

Colab reuses the checkout and virtual environment during a session. Start a fresh runtime to clone a different URL or load a newly uploaded ZIP. Recordings and finished transcripts remain in Drive when the runtime ends.
