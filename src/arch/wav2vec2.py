import torch
from ekko_bench.adapters import model
from ekko_bench.domain.config import SliceConfig
from transformers import AutoProcessor, Wav2Vec2ForCTC


class Wav2Vec2Model(model.BaseModel):
    def __init__(self,model_id: str, device: str= "cuda", use_adapter: bool = False):
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, device_map="balanced")
        self.decoder_map = {
            "yor" : "yor", 
            "ibo" : "ibo",
            "hau" : "hausa",
            "eng" : "english",
            "pcm" : "english"
            }
        self.use_adapter = use_adapter
        self.model.to(self.device)
        super().__init__(self.mo, device)

    def __call__(self, inputs, metadata: SliceConfig):
        lang = metadata.language
        if self.use_adapter:
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)

        inputs = self.processor(inputs, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcriptions = self.processor.decode(ids)
        return transcriptions