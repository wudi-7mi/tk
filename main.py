import asyncio

# ASR
import sounddevice as sd
import sherpa_ncnn
import soundfile as sf
import numpy as np
from scipy import signal

true_emotion = {
    "angry"     : 0,
    "happy"     : 0,
    "sad"       : 0,
    "surprise"  : 0,
    "neutral"   : 0,
    "fear"      : 0,
}


async def main():

    while True:
        i = 0
        await receive_wakeup_signal()

        text = await go_asr()
        
        # 没有识别到语音 进入休眠
        if not text:
            continue
        
        
        t_ser = asyncio.create_task(go_ser())
        res_dui = await go_dui()

        res_ser = await t_ser
        await cmp_emotion(v1=res_ser, v2=res_dui)

        await show_emotion_to_robot()
        i += 1
        print(i)


async def receive_wakeup_signal(name="recv_wakeup"):
    s = input('Press Enter To Wake Up:')
    return s
    ...


async def go_ser(name="go_ser"):
    print(f"{name}_start")
    await asyncio.sleep(2)
    print(f"{name}_end")
    return 1
    ...


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

            # 通过调节数字控制时间（不精准）
            # 逻辑：没有读取到语音并且过去5秒->结束；读取到语音并且过去2秒->结束
            if i == 50 and not result:
                break
            elif i == 20 and result:
                break

            if last_result != result:
                i = 0
                last_result = result
                print("\r{}".format(result), end="", flush=True)

    print(result)
    audio_res = np.concatenate(audio_res)
    audio_data_resampled = signal.resample(audio_res, int(len(audio_res) * 44100 / sample_rate))
    sf.write("output.wav", audio_data_resampled, 44100)

    return result


async def go_tts(text:str):
    print(f"go_tts_start")
    await asyncio.sleep(1)
    print(f"go_tts_end")
    return 1


async def go_dui(name="go_dui"):
    print(f"{name}_start")
    await asyncio.sleep(1)
    print(f"{name}_end")
    return 1
    ...


async def cmp_emotion(ser_emo, dui_emo, name="cmp_emotion"):
    
    if ser_emo.emo_type == dui_emo.emo_type:
        if ser_emo.value > 200:
            true_emotion[ser_emo.emo_type] = ser_emo.value
        else:
            true_emotion[ser_emo.emo_type] = 200
    else:
        ...
        
    print(true_emotion[ser_emo.emo_type])
    ...


async def show_emotion_to_robot(name="show_emo"):
    print(f"{name}_start")
    await asyncio.sleep(1)
    print(f"{name}_end")
    ...


if __name__ == "__main__":
    asyncio.run(main())