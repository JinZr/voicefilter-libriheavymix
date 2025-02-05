import argparse
import os

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from model.embedder import SpeechEmbedder
from model.model import VoiceFilter
from utils.audio import Audio
from utils.hparams import HParam


def main(args, hp, mixscp, enrollments_dict):
    with torch.no_grad():
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)["model"]
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        os.makedirs(args.out_dir, exist_ok=True)

        for key, item in tqdm(enrollments_dict.items()):
            for spkid, enroll_path in item:
                dvec_wav, _ = librosa.load(enroll_path, sr=8000)
                dvec_mel = audio.get_mel(dvec_wav)
                dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
                dvec = embedder(dvec_mel)
                dvec = dvec.unsqueeze(0)

                mixed_wav, _ = librosa.load(mix_dict[key], sr=8000)
                mag, phase = audio.wav2spec(mixed_wav)
                mag = torch.from_numpy(mag).float().cuda()

                mag = mag.unsqueeze(0)
                mask = model(mag, dvec)
                est_mag = mag * mask

                est_mag = est_mag[0].cpu().detach().numpy()
                est_wav = audio.spec2wav(est_mag, phase)

                out_path = os.path.join(args.out_dir, f"{key}_{spkid}.wav")

                sf.write(out_path, est_wav, samplerate=8000)


def enrollments_to_dict(file_path):
    res = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        key, _, spkid, enroll, index = line.strip().split(",")
        if key not in res:
            res[key] = list()
        res[key].append(
            (
                spkid,
                f"/star-data/rui/libriheavy_ovlp_src_reverb/dev_2spk/{enroll}/{index}.flac",
            )
        )
    return res


def lines_to_dict(file_path):
    res = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        key, value = line.strip().split()
        res[key] = value
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "-e",
        "--embedder_path",
        type=str,
        required=True,
        help="path of embedder model pt file",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="path of checkpoint pt file"
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, required=True, help="directory of output"
    )

    args = parser.parse_args()

    enrollments = (
        "/star-home/jinzengrui/data/LibriheavyCSS/dev_2spk_kaldi_fmt/enrollment"
    )
    mixscp = "/star-home/jinzengrui/data/LibriheavyCSS/dev_2spk_kaldi_fmt/mix.scp"
    enrollments_dict = enrollments_to_dict(enrollments)
    mix_dict = lines_to_dict(mixscp)

    hp = HParam(args.config)

    main(args, hp, mixscp, enrollments_dict)
