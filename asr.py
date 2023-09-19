import asyncio
import sounddevice as sd
import sherpa_ncnn
import soundfile as sf
import numpy as np
from scipy import signal

async def go_asr() -> str:
    devices = sd.query_devices()
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    recognizer = sherpa_ncnn.Recognizer(
        tokens          = "./ncnn/tokens.txt",
        encoder_param   = "./ncnn/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin     = "./ncnn/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param   = "./ncnn/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin     = "./ncnn/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param    = "./ncnn/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin      = "./ncnn/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads     = 4,
    )
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""

    audio_res = []
    
    i = 0

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            audio_res.append(samples)
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            i += 1

            if i == 50 and not result:
                break
            elif i == 20 and result:
                break

            if last_result != result:
                i = 0
                last_result = result
                # print("\r{}".format(result), end="", flush=True)

    print(result)
    audio_res = np.concatenate(audio_res)
    audio_data_resampled = signal.resample(audio_res, int(len(audio_res) * 44100 / sample_rate))
    sf.write("1.wav", audio_data_resampled, 44100)

    return result

if __name__ == "__main__":

    asyncio.run(go_asr())