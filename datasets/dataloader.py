import random
from random import choice

import librosa
import torch
from torch.utils.data import DataLoader, Dataset

from utils.audio import Audio


def create_dataloader(hp, args, train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()

        for dvec_mel, target_mag, mixed_mag in batch:
            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        return dvec_list, target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(
            dataset=VFDataset(hp, args, True),
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
            drop_last=True,
            sampler=None,
        )
    else:
        return DataLoader(
            dataset=VFDataset(hp, args, False),
            collate_fn=test_collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


class VFDataset(Dataset):
    def __init__(self, hp, args, train):

        def lines_to_dict(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                res[key] = value
            return res

        def lines_to_dict_spk2src(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                res[key] = value.split(",")
            return res

        def lines_to_dict_spk2spk(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                wavid, spkid = key.split("+")
                if wavid not in res:
                    res[wavid] = dict()
                    res[wavid][spkid] = value
                else:
                    res[wavid][spkid] = value
            return res

        def enrollments_to_dict(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, _, spkid, enroll, index = line.strip().split(",")
                res[key] = (
                    spkid,
                    f"/star-data/rui/libriheavy_ovlp_src_reverb/dev_2spk/{enroll}/{index}.flac",
                )
            return res

        self.hp = hp
        self.args = args
        self.train = train

        if self.train:
            self.dvec_list = lines_to_dict_spk2src(hp.form.spk2src)
            self.target_wav_list = lines_to_dict_spk2spk(hp.form.spk2spk)
            self.mix2spk = lines_to_dict(hp.form.mix2spk)
            self.mix2spk_keys = list(self.mix2spk.keys())
            self.mixed_wav_list = lines_to_dict(hp.form.input)
            assert len(self.dvec_list) != 0, "no training file found"
        else:
            self.enrollments = enrollments_to_dict(hp.dev.enrollments)
            self.enrollments_keys = list(self.enrollments.keys())
            self.target_wav_list = lines_to_dict_spk2spk(hp.dev.spk2spk)
            self.mix2spk = lines_to_dict(hp.dev.mix2spk)
            self.mixed_wav_list = lines_to_dict(hp.dev.input)

        self.audio = Audio(hp)

    def __len__(self):
        if self.train:
            return len(self.mix2spk_keys)
        else:
            return len(self.enrollments)

    def __getitem__(self, idx):
        if self.train:  # need to be fast
            mix_key = self.mix2spk_keys[idx]
            mixed_path = self.mixed_wav_list[mix_key]
            target_spk = self.mix2spk[mix_key]
            target_path = self.target_wav_list[mix_key][target_spk]
            dvec_path = choice(self.dvec_list[target_spk])

            dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
            dvec_mel = self.audio.get_mel(dvec_wav)
            dvec_mel = torch.from_numpy(dvec_mel).float()

            target_mag, _ = self.wav2magphase(target_path)
            mixed_mag, _ = self.wav2magphase(mixed_path)
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            return dvec_mel, target_mag, mixed_mag
        else:
            dvec_key = self.enrollments_keys[idx]
            dvec_path, spkid = self.enrollments[dvec_key]
            dvec_mel = self.audio.get_mel(dvec_path)
            dvec_mel = torch.from_numpy(dvec_mel).float()

            target_wav, _ = librosa.load(
                self.target_wav_list[dvec_key][spkid],
                sr=self.hp.audio.sample_rate,
            )
            mixed_wav, _ = librosa.load(
                self.mixed_wav_list[dvec_key],
                sr=self.hp.audio.sample_rate,
            )
            target_mag, _ = self.wav2magphase(self.target_wav_list[dvec_key][spkid])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[dvec_key])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        if self.train:
            wav, _ = librosa.load(path, sr=self.hp.audio.sample_rate)
            wav_dur = len(wav)
            dur = 3 * self.hp.audio.sample_rate
            start = random.randint(0, max(0, wav_dur - dur))
            wav = wav[start : start + dur]
        else:
            wav, _ = librosa.load(path, sr=self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
