"""官方风格预测桥。契约：`baselines.predict.predict(model, inputs) -> {"outputs": ...}`。

刻意不在本文件里 re-export 函数：否则包属性 `baselines.predict` 会被同名函数覆盖，
`from baselines import predict` 拿到的就成了函数而不是模块。用法一律写成：

    from baselines.predict import predict, predict_auto, load
"""
