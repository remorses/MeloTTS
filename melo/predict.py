import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Any
import torch
from openvoice.api import ToneColorConverter
import urllib.request
import soundfile
import base64
import numpy as np
import io

from cog import BaseModel, BasePredictor, Input, Path, ConcatenateIterator
from openvoice import se_extractor

from melo.api import TTS
import sys
import shutil
import zipfile




class Output(BaseModel):
    audio: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "mps"

        # RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -d /app/openvoice -o checkpoints_v2_0417.zip
        # RUN unzip /app/openvoice/checkpoints_v2_0417.zip
        # RUN rm -f /app/openvoice/checkpoints_v2_0417.zip
        # RUN mv /app/openvoice/checkpoints_v2 /app/openvoice/openvoice/checkpoints

        checkpoints_url = "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
        checkpoints_zip = "checkpoints_v2_0417.zip"
        checkpoints_dir = "checkpoints_v2"

        # Check if the checkpoints directory already exists
        if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
            # Check if the zip file already exists
            if not os.path.exists(checkpoints_zip):
                # Download the zip file
                urllib.request.urlretrieve(checkpoints_url, checkpoints_zip)

            # Extract the zip file
            with zipfile.ZipFile(checkpoints_zip, "r") as zip_ref:
                zip_ref.extractall()

            # Remove the zip file
            os.remove(checkpoints_zip)

            # Move the extracted directory to the desired location
            shutil.move(checkpoints_dir, os.path.join(os.getcwd(), checkpoints_dir))

        ckpt_converter = "checkpoints_v2/converter"

        output_dir = "outputs_v2"
        print(f"finished downloading checkpoints at {output_dir}")

        tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", device=self.device
        )
        tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        os.makedirs(output_dir, exist_ok=True)

        target_se, audio_name = se_extractor.get_se(
            "resources/reference_audio.wav", tone_color_converter, vad=True
        )
        self.target_se = target_se
        self.tone_color_converter = tone_color_converter
        # source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        
        speaker_key = "en-br"
        source_se = torch.load(
            f"checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
            map_location=self.device,
        )
        self.source_se = source_se

        self.models = {
            # "EN": TTS(language="EN", device=device),
            # "ES": TTS(language="ES", device=device),
            # "FR": TTS(language="FR", device=device),
            # "ZH": TTS(language="ZH", device=device),
            # "JP": TTS(language="JP", device=device),
            # "KR": TTS(language="KR", device=device),
        }

    def example(self):
        # text = "Hello, how are you?"
        src_path = "resources/child.wav"
        # Run the tone color converter
        

        encode_message = "i am very happy"
        print("Tone color conversion started...")
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path="resources/output.wav",
            message=encode_message,
        )
        print("Tone color conversion complete.")

    def predict(
        self,
        language: str = Input(
            choices=["EN", "ES", "FR", "ZH", "JP", "KR"], default="EN"
        ),
        speaker: str = Input(
            description="For EN, choose a speaker, for other langauges, leave it blank.",
            choices=["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default", "-"],
            default="EN-US",
        ),
        text: str = Input(
            default="The field of text-to-speech has seen rapid development recently."
        ),
        speed: float = Input(
            description="Speed of the output.", default=1.0, ge=0.1, le=10.0
        ),
    ) -> ConcatenateIterator[Output]:
        """Run a single prediction on the model"""
        speaker_ids = self.models[language].hps.data.spk2id
        speaker_id = speaker_ids[speaker] if language == "EN" else speaker_ids[language]
        out_path = "/tmp/out.wav"
        iterator = self.models[language].tts_iter(text, speaker_id, speed=speed)
        for audio in iterator:
            # convert to dataURL
            # audio = np.array(audio).astype(np.float32)
            wav = io.BytesIO()
            soundfile.write(
                wav, audio, self.models[language].hps.data.sampling_rate, format="wav"
            )

            data = "data:audio/wav;base64," + base64.b64encode(wav.getvalue()).decode()
            yield Output(audio=data)


if __name__ == "__main__":
    pred = Predictor()
    pred.setup()
    pred.example()
