from ekko_bench.adapters import model
from ekko_bench.domain.config import SliceConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WHISPER_CONFIG = {
    "small" :  "openai/whisper-small",
    "medium" : "openai/whisper-medium",
    "large" : "openai/whisper-large",
}


class Whisper(model.BaseModel):
    def __init__(self, model_id: str, device: str="cuda"):
        self.device = device
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, device_map="balanced")
        self.decoder_map = {
            "yor" : "yoruba", 
            "ibo" : "yoruba",
            "hau" : "hausa",
            "eng" : "english",
            "pcm" : "english"
            }

        self.model.to(self.device)
        super().__init__(model_id, device)

    def __call__(self, inputs, metadata: SliceConfig):
        lang = metadata.language
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(task="transcribe", language=self.decoder_map[lang])
        input_features = self.processor(inputs, sampling_rate=16_000, return_tensors='pt').input_features.to(self.device)
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcriptions = self.processor.decode(predicted_ids, skip_special_tokens=True)
        return transcriptions