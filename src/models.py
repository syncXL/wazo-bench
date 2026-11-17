from collections import namedtuple

from . import loader

ModelCard = namedtuple('ModelCard', ['id', 'link', 'caller', 'notes'])

mms = ModelCard(
    id= "facebook/mms-1b-all",
    link="https://huggingface.co/facebook/mms-1b-all",
    caller=loader.load_mms,
    notes=""
)

omniasr_1b = ModelCard(
    id="aadel4/omniASR-CTC-1B-v2",
    link="https://huggingface.co/aadel4/omniASR-CTC-1B-v2",
    caller=loader.load_omniasr_1b,
    notes="This is the converted version of OmniAsr from fairseq2, this is because I'm using Windows not Linux"
)

omniasr_3b = ModelCard(
    id="nolimitsxl/omniASR-CTC-3B",
    link="https://huggingface.co/nolimitsxl/omniASR-CTC-3B",
    caller=loader.load_omniasr_3b,
    notes="This is the converted version of OmniAsr from fairseq2, this is because I'm using Windows not Linux"
)

whisper_large = ModelCard(
    id="openai/whisper-large-v3",
    link="https://huggingface.co/openai/whisper-large-v3",
    caller=loader.load_whisper_large,
    notes="Whisper do not have Igbo and Pidgin token, for those languages, it uses Yoruba and English token respectively"
)

eko_stt = ModelCard(
    id="munasco/eko-stt-v1",
    link="https://huggingface.co/munasco/eko-stt-v1",
    caller=loader.load_eko_stt,
    notes="Whisper do not have Igbo and Pidgin token, for those languages, it uses Yoruba and English token respectively"
)

sbpn = ModelCard(
    id="ogunlao/SBPN_multilingual_large",
    link="https://huggingface.co/ogunlao/SBPN_multilingual_large",
    caller=loader.load_sbpn,
    notes=""
)


MODELS = {i.id:i for i in [mms, omniasr_1b, omniasr_3b, whisper_large, eko_stt, sbpn]}