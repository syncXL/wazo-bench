import nemo.collections.asr as nemo_asr
from ekko_bench.adapters import model
from ekko_bench.domain.config import SliceConfig


class Parakeet(model.BaseModel):
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(model_id,map_location=device)
        self.model.eval()
        super().__init__(model_id=model_id, device=device)

    def __call__(self, inputs, metadata: SliceConfig):
        transcripts = self.model.transcribe(inputs,batch_size=16)
        return transcripts
