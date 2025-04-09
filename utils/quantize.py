# utils/quantize.py
import torch

def quantize_model(model, example_input_src, example_input_tgt):
    model.eval()
    model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        model(example_input_src, example_input_tgt)  # Calibration step
    torch.quantization.convert(model, inplace=True)
    return model
