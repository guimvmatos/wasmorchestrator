import torchvision.models as models
import torch

model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "vgg11_bn.onnx", opset_version=13)