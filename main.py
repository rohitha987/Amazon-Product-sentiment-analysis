import pandas as pd
import numpy as np
absa=pd.read_json("atepc_inference.result.json")
absa1=absa[['aspect','sentiment','confidence']]
absa_dict=absa1.to_dict()
asps=list(absa_dict['aspect'].values())
sents=list(absa_dict['sentiment'].values())
conf=list(absa_dict['confidence'].values())
asps_new= list(np.concatenate(asps))
sents_new= list(np.concatenate(sents))
conf_new= list(np.concatenate(conf))
print(asps_new)
print(sents_new)
print(conf_new)
for i in range(len(asps_new)):
    if sents_new[i]=='Negative':
        conf_new[i]=conf_new[i]*(-1)
print(conf_new)