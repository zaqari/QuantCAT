import pandas as pd
import numpy as np

# If on the remote server
from kgen2.BERTcruncher.midBERT_gpu import *

level = 7
mod = embeds(device='cuda')
output_marker ='-sum'+str(level)


W_ = ['immigration', 'immigrants', 'wall', 'border', 'mexico']

################################################################
### Political corpus
################################################################
PATH = '/home/zaq/d/poli/'
output_name = '538vecs'+output_marker+'.csv'
dataset = 'corpora/538senators.csv'


df = pd.read_csv(PATH + dataset, encoding='ISO-8859-1')
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(1) Set up .csv file for data repo
data = pd.DataFrame(columns=['party', 'user', 'date', 'w', 'vec', 'text'])
data.to_csv(PATH+output_name, index=False, encoding='utf-8')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    p, u, text, d = df[['party', 'user', 'text', 'created_at']].loc[k]
    #print(text, '\n')
    if sum([wi in text.lower() for wi in W_]) > 0:
        for w in W_:
            try:
                vecs = mod(w, str(text).lower(), level=level)
                update = [[p, u, d, w, str(vec.view(-1).cpu().tolist()), k] for vec in vecs]
                update = pd.DataFrame(np.array(update), columns=list(data))
                update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
            except ValueError:
                0
            except IndexError:
                0
            except RuntimeError:
                0





################################################################
### BLM corpus
################################################################
PATH = '/home/zaq/d/poli/'
output_name = 'BLMvecs'+output_marker+'.csv'
dataset = 'corpora/blm.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['tweet_text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(1) Set up .csv file for data repo
data = pd.DataFrame(columns=['party', 'user', 'date', 'w', 'vec', 'text'])
data.to_csv(PATH+output_name, index=False, encoding='utf-8')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    u, text, d = df[['tweet_id', 'text', 'tweet_created_dt']].loc[k]
    # print(text, '\n')
    if sum([wi in text.lower() for wi in W_]) > 0:
        for w in W_:
            try:
                vecs = mod(w, str(text).lower(), level=level)
                update = [['BLM', 'BLM', d, w, str(vec.view(-1).cpu().tolist()), k] for vec in vecs]
                update = pd.DataFrame(np.array(update), columns=list(data))
                update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
            except ValueError:
                0
            except IndexError:
                0
            except RuntimeError:
                0




################################################################
### OBA corpus
################################################################
PATH = '/home/zaq/d/poli/'
output_name = 'OBAvecs'+output_marker+'.csv'
dataset = 'corpora/obama-2.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['Tweet-text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(1) Set up .csv file for data repo
data = pd.DataFrame(columns=['party', 'user', 'date', 'w', 'vec', 'text'])
data.to_csv(PATH+output_name, index=False, encoding='utf-8')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','Date']].loc[k]
    #print(text, '\n')
    if sum([wi in text.lower() for wi in W_]) > 0:
        for w in W_:
            try:
                vecs = mod(w, str(text).lower(), level=level)
                update = [['OBA', 'OBA', d, w, str(vec.view(-1).cpu().tolist()), k] for vec in vecs]
                update = pd.DataFrame(np.array(update), columns=list(data))
                update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
            except ValueError:
                0
            except IndexError:
                0
            except RuntimeError:
                0





################################################################
### TRM corpus
################################################################
PATH = '/home/zaq/d/poli/'
output_name = 'TRMvecs'+output_marker+'.csv'
dataset = 'corpora/trump-2.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(1) Set up .csv file for data repo
data = pd.DataFrame(columns=['party', 'user', 'date', 'w', 'vec', 'text'])
data.to_csv(PATH+output_name, index=False, encoding='utf-8')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','date']].loc[k]
    #print(text, '\n')
    if sum([wi in text.lower() for wi in W_]) > 0:
        for w in W_:
            try:
                vecs = mod(w, str(text).lower(), level=level)
                update = [['TRM', 'TRM', d, w, str(vec.view(-1).cpu().tolist()), k] for vec in vecs]
                update = pd.DataFrame(np.array(update), columns=list(data))
                update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
            except ValueError:
                0
            except IndexError:
                0
            except RuntimeError:
                0




################################################################
### HRC corpus
################################################################
PATH = '/home/zaq/d/poli/'
output_name = 'HRCvecs'+output_marker+'.csv'
dataset = 'corpora/HillaryClintonTweets.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(1) Set up .csv file for data repo
data = pd.DataFrame(columns=['party', 'user', 'date', 'w', 'vec', 'text'])
data.to_csv(PATH+output_name, index=False, encoding='utf-8')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','date']].loc[k]
    #print(text, '\n')
    if sum([wi in text.lower() for wi in W_]) > 0:
        for w in W_:
            try:
                vecs = mod(w, str(text).lower(), level=level)
                update = [['HRC', 'HRC', d, w, str(vec.view(-1).cpu().tolist()), k] for vec in vecs]
                update = pd.DataFrame(np.array(update), columns=list(data))
                update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
            except ValueError:
                0
            except IndexError:
                0
            except RuntimeError:
                0




# ################################################################
# ### Shutova corpus
# ################################################################
# PATH = '/home/zaq/d/shutova/'
# df = pd.read_csv(PATH + 'metaphor-corpus-READABLE.csv')
# df['sID'] = range(len(df))
#
# outpath = 'vecs'+output_marker+'.csv'
#
#
# print(list(df))
# print(PATH, len(df), '\n')
#
# #(1) Set up .csv file for data repo
# data = pd.DataFrame(columns=['ID', 'cm', 'source', 'target', 'schema.slot', 'schema', 'vec', 'text'])
# data.to_csv(PATH+outpath, index=False, encoding='utf-8')
#
# #(2) Generate embeddings with appropriate metadata
# for CM, source, target, slot, schema, text, sID in df[['Source CM', 'Source LM', 'Target LM', 'Schema Slot', 'Schema', 'Sentence', 'sID']].values:
#     #print(text)
#     try:
#         vecs = mod(str(target).lower(), str(text).lower(), level=level)
#         update = [[sID, CM, source, target, slot, schema, str(vec.view(-1).cpu().tolist()), str(text)] for vec in vecs]
#         update = pd.DataFrame(np.array(update), columns=list(data))
#         update.to_csv(PATH + outpath, index=False, encoding='utf-8', header=False, mode='a')
#     except ValueError:
#         0
#     except IndexError:
#         0
#     except RuntimeError:
#         0
#
#
#
# ################################################################
# ### Mohler corpus
# ################################################################
# PATH = '/home/zaq/d/mohler/'
# output_name = 'vecs'+output_marker+'.csv'
# dataset = 'mohler-texts.csv'
#
#
# df = pd.read_csv(PATH + dataset, encoding='utf-8')
#
# # print('pre-dropping of duplicates', len(df))
# # df = df.drop_duplicates(subset=['LmTarget','text'])
# # df.index=range(len(df))
#
# print(list(df))
# print(PATH+dataset, len(df), '\n')
#
# #(1) Set up .csv file for data repo
# data = pd.DataFrame(columns=['ID', 'cm', 's', 'w', 'vec', 'text'])
# data.to_csv(PATH+output_name, index=False, encoding='utf-8')
#
# #(2) Generate embeddings with appropriate metadata
# for k in df.index:
#     ID,cm,s,w,text = df[['ID', 'cm', 'LmSource', 'LmTarget', 'text']].loc[k]
#     #print(text, '\n')
#     try:
#         new_term = w.replace(' ', '')
#         vecs = mod(new_term.lower(), str(text).replace(w,new_term).lower(), level=level)
#         update = [[ID,cm,s,w,str(vec.view(-1).cpu().tolist()), text] for vec in vecs]
#         update = pd.DataFrame(np.array(update), columns=list(data))
#         update.to_csv(PATH+output_name, index=False, encoding='utf-8', header=False, mode='a')
#     except ValueError:
#         0
#     except IndexError:
#         0
#     except RuntimeError:
#         0
