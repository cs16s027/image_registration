from gym.envs.registration import register

register(
    id = 'imgreg_train-v0',
    entry_point='imgreg.envs:ImgRegTrain',
)

register(
    id = 'imgreg_train-v1',
    entry_point='imgreg.envs:ImgRegTrainv1',
)

register(
    id = 'imgreg_train-v2',
    entry_point='imgreg.envs:ImgRegTrainv2',
)

register(
    id = 'imgreg_train-v3',
    entry_point='imgreg.envs:ImgRegTrainv3',
)

register(
    id = 'imgreg_train-v4',
    entry_point='imgreg.envs:ImgRegTrainv4',
)

register(
    id = 'imgreg_test-v0',
    entry_point='imgreg.envs:ImgRegTest',
)

register(
    id = 'imgreg_test-v1',
    entry_point='imgreg.envs:ImgRegTestv1',
)

register(
    id = 'imgreg_test-v2',
    entry_point='imgreg.envs:ImgRegTestv2',
)

register(
    id = 'imgreg_test-v3',
    entry_point='imgreg.envs:ImgRegTestv3',
)
