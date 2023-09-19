import asyncio

import sounddevice as sd
import soundfile as sf

from paddlespeech.cli.tts.infer import TTSExecutor
tts = TTSExecutor()
tts(
    text='对数据集进行预处理',
    output='output.wav',
    am='fastspeech2_csmsc',
    voc='hifigan_csmsc',
    lang='zh',
    use_onnx=True,
    cpu_threads=4)

async def go_tts(text:str):
    output_file = "output_tts.wav"
    tts(
        text='对数据集进行预处理',
        output=output_file,
        am='fastspeech2_csmsc',
        voc='hifigan_csmsc',
        lang='zh',
        use_onnx=True,
        cpu_threads=4)
    print("yes")
    audio_data, sample_rate = sf.read(output_file)
    sd.play(audio_data, sample_rate)
    await sd.wait()

    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(go_tts("你觉得怎么样？"))