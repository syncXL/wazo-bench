from ekko_bench.adapters import model

from src import arch


def load_mms(device: str) -> model.BaseModel:
    return arch.Wav2Vec2Model("facebook/mms-1b-all", device= device, use_adapter=True)


def load_omniasr_1b(device: str) -> model.BaseModel:
    return arch.Wav2Vec2Model("aadel4/omniASR-CTC-1B-v2", device=device, use_adapter=False)

def load_omniasr_3b(device: str) -> model.BaseModel:
    return arch.Wav2Vec2Model("nolimitsxl/omniASR-CTC-3B", device=device, use_adapter=False)


def load_whisper_large(device: str) -> model.BaseModel:
    return arch.Whisper("openai/whisper-large-v3", device=device)

def load_eko_stt(device: str) -> model.BaseModel:
    eko_stt = arch.Whisper("munasco/eko-stt-v1", device=device)
    eko_stt.decoder_map = {
        "eng" : "en",
        "yor" : "yo",
        "ibo" : "yo",
        "ha" : "ha",
        "pcm" : "en"
    }
    return eko_stt

def load_sbpn(device: str) -> model.BaseModel:
    return arch.Parakeet("ogunlao/SBPN_multilingual_large", device=device)

