import torch
import sentencepiece as spm
from gpt_model import Transformer, ModelConfig
from utils import generate
import socket

def send_message(sock, message, syslog_server="127.0.0.1", port=10514):
    syslog_message = f"{message}"
    try:
        sock.sendto(syslog_message.encode('utf-8'), (syslog_server, port))
    except Exception as e:
        print(f"Failed to send message: {e}")
    finally:
        sock.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece_bpe.model")


# Define the model
my_config = ModelConfig()
print(f"model config: {my_config}")
model = Transformer(my_config).to(device)
print(f"model #params: {sum(p.numel() for p in model.parameters())}")
model.load_state_dict(torch.load("model.pth"))
model.eval()


example_of_start = torch.tensor([1]).reshape(1,-1)
example_of_start.to(device)
model.to(device) 
while True:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gen_ids = generate(model, device, example_of_start, max_new_tokens=50, do_sample=True, temperature=1, top_k=20).reshape(-1).tolist()
    res = sp.DecodeIds(gen_ids)
    print(res)
    send_message(sock, res)