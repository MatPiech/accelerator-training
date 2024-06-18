from pathlib import Path
from time import perf_counter

import click
import onnxruntime
import numpy as np
from tqdm import tqdm


@click.command()
@click.option('--model-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--provider', type=click.Choice(['cpu', 'hailo']), default='cpu')
@click.option('--iterations', type=int, default=10000)
def main(model_path: Path, provider: str, iterations: int):
    if provider == 'hailo':
        ep_list = ['HailoExecutionProvider']
    else:
        ep_list = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=ep_list)
    print('Session inputs:', [(si.name, si.shape) for si in session.get_inputs()])
    print('Session outputs:',[(so.name, so.shape) for so in session.get_outputs()])
    input_dict = {
        si.name: np.random.randn(1, *si.shape[1:]).astype(np.float32) for si in session.get_inputs()
    }
    inference_start = perf_counter()
    for _ in tqdm(range(iterations)):
        outputs = session.run([output.name for output in session.get_outputs()], input_dict)
    inference_time = perf_counter() - inference_start
    print(f'Average inference time: {inference_time/iterations*1000:.3f} ms')
    print(f'Inference outputs shape: {outputs[0].shape}')


if __name__ == "__main__":
    main()
