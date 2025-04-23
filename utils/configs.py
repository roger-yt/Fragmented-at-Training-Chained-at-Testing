from transformers import GPT2Config

class MyGPT2Config(GPT2Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_type = kwargs.get('network_type', None)
        self.low_rank = kwargs.get('low_rank', None)
class MyMLPConfig:
    def __init__(self, *args, **kwargs):
        self.window_size = kwargs.get('window_size', None)
        self.in_dim = kwargs.get('in_dim', None)
        self.out_dim = kwargs.get('out_dim', None)
        self.hidden_dim = kwargs.get('hidden_dim', None)
        self.layer_num = kwargs.get('layer_num', None)
        self.all_token_ids = kwargs.get('all_token_ids', None)
        self.vocab_size = kwargs.get('vocab_size', None)