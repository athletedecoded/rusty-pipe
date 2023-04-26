import torch
import torch.onnx as onnx
from torch.utils.mobile_optimizer import optimize_for_mobile

print("Loading model.pt...")
model_pt = torch.jit.load("model.pt")
model_pt.eval()
scripted_model = torch.jit.script(model_pt)

print("Optimizing for mobile...")
optimized_model = optimize_for_mobile(scripted_model)

print("Model saved to model.ot")
optimized_model.save("model.ot")

# Specify an input tensor to the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
onnx.export(model, dummy_input, "model.onnx")