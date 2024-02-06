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

        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        if self.train:
            self.dvec_list = lines_to_dict_spk2src(hp.form.spk2src)
            self.target_wav_list = lines_to_dict_spk2spk(hp.form.spk2spk)
            self.mix2spk = lines_to_dict(hp.form.mix2spk)
            self.mix2spk_keys = self.mix2spk.keys()
            self.mixed_wav_list = lines_to_dict(hp.form.input)
        else:
            self.dvec_list = lines_to_dict_spk2src(hp.dev.spk2src)
            self.target_wav_list = lines_to_dict_spk2spk(hp.dev.spk2spk)
            self.mix2spk = lines_to_dict(hp.dev.mix2spk)
            self.mix2spk_keys = self.mix2spk.keys()
            self.mixed_wav_list = lines_to_dict(hp.dev.input)

        assert len(self.dvec_list) != 0, "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.mix2spk_keys)

    def __getitem__(self, idx):

        mix_key = self.mix2spk_keys[idx]
        mixed_path = self.mixed_wav_list[mix_key]
        target_spk = self.mix2spk[mix_key]
        target_path = self.target_wav_list[mix_key][target_spk]
        dvec_path = choice(self.dvec_list[target_spk])

        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train:  # need to be fast
            target_mag, _ = self.wav2magphase(target_path)
            mixed_mag, _ = self.wav2magphase(mixed_path)
            return dvec_mel, target_mag, mixed_mag
        else:
            target_wav, _ = librosa.load(
                self.target_wav_list[idx], self.hp.audio.sample_rate
            )
            mixed_wav, _ = librosa.load(
                self.mixed_wav_list[idx], self.hp.audio.sample_rate
            )
            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
