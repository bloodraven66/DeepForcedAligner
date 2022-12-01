import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
import yaml


def read_metafile(path: str, folder, dur_path) -> Dict[str, str]:
    text_dict = {}
    txt_files = []
    audio_files = []
    print(path)
    for filename in os.listdir(folder):
        if filename.startswith(str(path)):
            txt_files.extend(get_files(os.path.join(folder, filename), '.txt'))
            audio_files.extend(get_files(os.path.join(folder, filename), '.wav'))
    for textfile in txt_files:
        with open(str(textfile), 'r') as f:
            line = f.read()
        text_dict[textfile.stem] = line

    mapping = {
                'Hindi_M':'dur_hi_m',
                'Hindi_F':'dur_hi_f',
                'Telugu_M':'dur_te_m',
                'Telugu_F':'dur_te_f',
                'Marathi_M':'dur_mr_m',
                'Marathi_F':'dur_mr_f',
                }
    with open(os.path.join(dur_path, mapping[path.stem]), 'r') as f:
        lines = f.read().split('\n')[:-1]
    lines = set([Path(l.split('\t')[0]).stem for l in lines if float(l.split('\t')[-1]) > 2])
    text_dict = {t:text_dict[t] for t in text_dict if t in lines}
    return text_dict, audio_files


def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension='.wav') -> List[Path]:
    return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)


def to_device(batch: dict, device: torch.device) -> tuple:
    tokens, mel, tokens_len, mel_len = batch['tokens'], batch['mel'], \
                                       batch['tokens_len'], batch['mel_len']
    tokens, mel, tokens_len, mel_len = tokens.to(device), mel.to(device), \
                                       tokens_len.to(device), mel_len.to(device)
    return tokens, mel, tokens_len, mel_len