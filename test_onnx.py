from funasr_onnx import Paraformer
# model_dir = "output/"
# model = Paraformer(model_dir, batch_size=1, quantize=True)

model_dir = "/home/niwang/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['创建警单.wav']

result = model(wav_path)
print(result)