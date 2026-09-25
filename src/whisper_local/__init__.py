from faster_whisper import WhisperModel

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float16",
)

segments, info = model.transcribe(
    "/home/shogo/projects/tomody/d-sha/wavs_ja/01_disp_hmi_scene.wav",
    language="ja",
)

for segment in segments:
    print(segment.text)
