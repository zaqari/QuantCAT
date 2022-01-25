import pandas as pd
import numpy as np

# from shared.mod.BERTattn import *
# PATH = 'OpenMetaphor/data/'
# output_name = 'attention-weights-check.csv'
# dataset = 'source-webster.csv'

# If on the remote server
from kgen2.BERTcruncher.BERTattn import *

level = 7
PATH = '/home/zaq/d/poli/'
output_name = 'poli-attn-lvl'+str(level)+'.csv'

mod = attention()

# W_ = ['immigration', 'immigrants']
W_ = ['immigration', 'immigrants', 'wall', 'border', 'mexico']

# (1) Set up .csv file for data repo
data = pd.DataFrame(columns=['line.no', 'user', 'party', 'date', 'w', 'attention', 'tokens'])
data.to_csv(PATH + output_name, index=False, encoding='utf-8')


##################################################################################################
##### OBA Corpus
##################################################################################################
dataset = 'obama-2.csv'

df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['Tweet-text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(PATH+dataset, len(df), '\n')

# (2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','Date']].loc[k]
    # print(text, '\n')
    for w in W_:
        try:
            attn, toks, _ = mod(w.lower(), str(text).lower(), level=level)
            update = [[k, 'OBA', 'OBA', d, w, str(attn.detach().view(-1).tolist()), str(toks.tolist())]]
            update = pd.DataFrame(np.array(update).reshape(-1,len(list(data))), columns=list(data))
            update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
        except ValueError:
            0
        except IndexError:
            0
        except RuntimeError:
            0


##################################################################################################
##### TRM Corpus
##################################################################################################
dataset = 'trump-2.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

# (2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','date']].loc[k]
    # print(text, '\n')
    for w in W_:
        try:
            attn, toks, _ = mod(w.lower(), str(text).lower(), level=level)
            update = [[k, 'TRM', 'TRM', d, w, str(attn.detach().view(-1).tolist()), str(toks.tolist())]]
            update = pd.DataFrame(np.array(update).reshape(-1,len(list(data))), columns=list(data))
            update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
        except ValueError:
            0
        except IndexError:
            0
        except RuntimeError:
            0

################################################################
### Political corpus
################################################################
PATH = '/home/zaq/d/poli/'
dataset = '538senators.csv'


df = pd.read_csv(PATH + dataset, encoding='ISO-8859-1')
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

#(2) Generate embeddings with appropriate metadata
for k in df.index:
    p, u, text, d = df[['party', 'user', 'text','created_at']].loc[k]
    #print(text, '\n')
    for w in W_:
        try:
            attn, toks, _ = mod(w.lower(), str(text).lower(), level=level)
            update = [[k, u, p, d, w, str(attn.detach().view(-1).tolist()), str(toks.tolist())]]
            update = pd.DataFrame(np.array(update).reshape(-1, len(list(data))), columns=list(data))
            update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
        except ValueError:
            0
        except IndexError:
            0
        except RuntimeError:
            0




##################################################################################################
##### HRC Corpus
##################################################################################################
dataset = 'HillaryClintonTweets.csv'


df = pd.read_csv(PATH + dataset)
print('pre-dropping of duplicates', len(df))
df['text'] = [str(entry).split('http')[0] for entry in df['text'].values]
df = df.loc[df['text'].isin([text for text in df['text'].values if sum([wi in str(text).lower() for wi in W_]) > 0])]
df = df.drop_duplicates(subset=['text'])
# df.index=range(len(df))

print(list(df))
print(W_)
print(PATH+dataset, len(df), '\n')

# (2) Generate embeddings with appropriate metadata
for k in df.index:
    text, d = df[['text','date']].loc[k]
    # print(text, '\n')
    for w in W_:
        try:
            attn, toks, _ = mod(w.lower(), str(text).lower(), level=level)
            update = [[k, 'HRC', 'HRC', d, w, str(attn.detach().view(-1).tolist()), str(toks.tolist())]]
            update = pd.DataFrame(np.array(update).reshape(-1,len(list(data))), columns=list(data))
            update.to_csv(PATH + output_name, index=False, encoding='utf-8', header=False, mode='a')
        except ValueError:
            0
        except IndexError:
            0
        except RuntimeError:
            0
