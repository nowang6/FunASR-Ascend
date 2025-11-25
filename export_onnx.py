from funasr import AutoModel

model = AutoModel(model="paraformer-zh-streaming")

res = model.export(quantize=False, opset_version=18)