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
    id = 'imgreg_test-v0',
    entry_point='imgreg.envs:ImgRegTest',
)
