from . import rosa as lf
from . import models
from . import utils

def predict(config, audio_path: str, model) -> dict:
    """
    预测音频情感

    Args:
        config: 配置项
        audio_path (str): 要预测的音频路径
        model: 加载的模型
    """

    # utils.play_audio(audio_path)

    test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', config.class_labels[int(result)])
    ["angry", "fear", "happy", "neutral", "sad", "surprise"]
    # print('Probability: ', result_prob)
    return {
        "emotion": config.class_labels[int(result)],
        "verbose": {
            "angry":    result_prob[0],
            "fear":     result_prob[1],
            "happy":    result_prob[2],
            "neutral":  result_prob[3],
            "sad":      result_prob[4],
            "suprise":  result_prob[5],
        }
    }


def infer(audio_path='./output.wav') -> dict:
    config = utils.parse_opt()
    model = models.load(config)
    return predict(config, audio_path, model)


if __name__ == '__main__':
    audio_path = '../output.wav'

    config = utils.parse_opt()
    model = models.load(config)
    predict(config, audio_path, model)
