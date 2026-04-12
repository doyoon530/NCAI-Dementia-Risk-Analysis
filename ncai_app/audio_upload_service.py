import os
import uuid

from werkzeug.utils import secure_filename


def save_audio_upload(audio_file, upload_dir: str, allowed_extensions: set[str]) -> str:
    original_name = secure_filename(audio_file.filename) or "audio.wav"
    extension = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""

    if extension not in allowed_extensions:
        return ""

    unique_name = f"{uuid.uuid4()}_{original_name}"
    file_path = os.path.join(upload_dir, unique_name)
    audio_file.save(file_path)
    return file_path
