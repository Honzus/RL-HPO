from envs.registration import register,make
# Metadataset
# ----------------------------------------       

for metadata in ['nnMeta-v4','svmMeta-v4']:
    if metadata == 'nnMeta-v4':
        obs = (25,50);metafeatures=21;hyp=3 #17,13,3
        dtemp = 'nn-meta'
    else:
        obs = (10,50);metafeatures=3;hyp=6
        dtemp = 'svm-meta'
    for fold in range(5):
        register(
            id=metadata+str(fold),
            entry_point='envs.hylap:nnMetaEnv',
            kwargs={'configs' : {'obs_space':obs,
                                 'obs_type':'multi_variate',
                                 'num_datasets':5,
                                 'fold':fold,
                                 'nfolds':5,
                                 'path':'metadata/{}/split-{}/'.format(dtemp,fold),
                                 'metafeatures':metafeatures,
                                 'hyperparameters':hyp,
                                 'seed':0}},
        )