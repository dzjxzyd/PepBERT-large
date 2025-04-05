All pretrained models in this repository have been uploaded to Hugging Face for easy access. Usage tutorials are provided in the respective README.md files on each model’s Hugging Face page.

You can access the models at the following links:

	•	[PepBERT-large-UniParc](https://huggingface.co/dzjxzyd/PepBERT-large-UniParc)
 
 	•	[PepBERT-large-UniRef100](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef100)
  
  	•	[PepBERT-large-UniRef90](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef90)
   
   	•	[PepBERT-large-UniRef50](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef50)

### A example tutorial for usage in your custom peptide embeddings.
```
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from model import build_transformer
from config import get_config

# 1) Download tokenizer.json
tokenizer_path = hf_hub_download(repo_id="dzjxzyd/PepBERT-large-UniRef50", filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

# 2) Download model weights tmodel_16.pt
weights_path = hf_hub_download(repo_id="dzjxzyd/PepBERT-large-UniRef50", filename="tmodel_16.pt")

# 3) Initialize the model structure and load the weights
device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
config = get_config()
model = build_transformer(
    src_vocab_size=tokenizer.get_vocab_size(),
    src_seq_len=config["seq_len"],
    d_model=config["d_model"]
)
state = torch.load(weights_path, map_location=torch.device(device))
model.load_state_dict(state["model_state_dict"])
model.eval()

# 4) Generate for your embeddings 
# Example amino acid sequence
sequence = "KRKGFLGI"

# tokenization of the input sequences
encoded_ids = [tokenizer.token_to_id("[SOS]")] + tokenizer.encode(sequence).ids + [tokenizer.token_to_id("[EOS]")]
# 3) Convert to a tensor of shape [1, seq_len]
input_ids = torch.tensor([encoded_ids], dtype=torch.int64)

# 4) Forward through the model
with torch.no_grad():
    encoder_mask = torch.ones((1, 1, 1, input_ids.size(1)), dtype=torch.int64)
    emb = model.encode(input_ids, encoder_mask)
    # Remove the first token ([SOS]) and the last token ([EOS]) from the sequence dimension
    emb_no_first_last = emb[:, 1:-1, :]  # shape: [1, seq_len-2, d_model]
    emb_avg = emb_no_first_last.mean(dim=1)  # shape: [1, d_model]
    
print("Shape of emb_avg:", emb_avg.shape)
print("emb_avg:", emb_avg)
```
