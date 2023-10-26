import onnxruntime as ort
import numpy as np

def load_session(path: str) -> ort.InferenceSession:
    # Create a list with only the CUDA execution provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(path, providers=providers)
    return session

def infer(inference_session: ort.InferenceSession, input_data: np.array) -> np.array:
    input_name = inference_session.get_inputs()[0].name
    output_name = inference_session.get_outputs()[0].name
    inference_inputs = {input_name: input_data}
    outputs = inference_session.run(
        [output_name], 
        inference_inputs
        )
    return outputs[0]
