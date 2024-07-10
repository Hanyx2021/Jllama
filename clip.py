import torch
import jittor as jt

src_path = "Meta-Llama-3-8B/consolidated.00.pth"
dst_path = "Meta-Llama-3-8B/consolidated.00.pkl"

clip = torch.load(src_path)

for k in clip.keys():
    print(k)
    clip[k] = clip[k].float().cpu()

jt.save(clip,dst_path)