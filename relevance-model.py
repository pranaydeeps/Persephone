from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import gensim


import tweepy
import os

#Access Tokens Here#
consumer_key = 'j0yxfNvNMib54TZh1X41E5bRW'
consumer_secret = 'TocWM8XOXYzOKrlL0kcwjdbHGgPv66Mg030rfYQfoYsTyDue2P'
access_token = '839292451-tpqqSZTPUAojpOPzJZU1SlUd7VJpMlqXnhhEOX00'
access_token_secret = 'GCAGCQWR9O2Fy1ESEb0DEjSWxUsDPuuHXmixldsUQ0p6q'

#GET API ACCESS#
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Request for user, if found proceeed, else quit#
# handle = raw_input('Query handle? \t')
# try:
#     user = api.get_user(handle)
# except:
#     print 'GG. Quiting.'
#     import sys
#     sys.exit(0)
# print 'Found User: ' + str(user.screen_name) + ' with Follower Count:' + str(user.followers_count)


# analysis_texts = ''
# for status in user.timeline(count=20):
#     analysis_texts+= ' ' + status.text

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

import numpy as np
data = np.load('tweets-qoruz.npy')

current_id = data[0][0]
temp_list = ''
new_data = []
for point in data:
    if point[0] == current_id:
        temp_list = temp_list + ' ' + point[1]
    else:
        new_data.append([current_id,temp_list])
        temp_list = ''
        temp_list = temp_list + ' ' + point[1]
        current_id = point[0]

# new_data.append([user.id,analysis_texts])
new_data = np.asarray(new_data)
doc_complete = new_data[:,1]
twitter_ids = new_data[:,0]

print(len(new_data))
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and i.startswith('#')!=True and i.startswith('http')!=True])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import unidecode
doc_clean = [clean(unidecode.unidecode(doc)).split() for doc in doc_complete]
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
from gensim.models.doc2vec import LabeledSentence

ids=0
labeled_docs = []
for doc in doc_clean:
    labeled_docs.append(LabeledSentence(doc,[twitter_ids[ids]]))
    ids+=1

model = gensim.models.Doc2Vec(size=20, window=5, min_count=8, workers=4,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(labeled_docs)

it = labeled_docs

print 'Training Model Now'

for epoch in range(10):
    model.train(it, total_examples=7162, epochs=1)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it, total_examples=7162,epochs=1)

print(model.docvecs.most_similar([str(user.id)]))



import json
from pymongo import MongoClient
import numpy as np


with open('geolocs.npy') as geofile:
    geolocs = np.load(geofile)

geolocs_dict = {}
for geoloc in geolocs:
    geolocs_dict[list(geoloc.keys())[0]] = list(geoloc.values())[0]

def replace_none(point):
    if 'N/' in str(point):
        return 0
    if point is None:
        return 0
    else:
        if 'K' in str(point):
            point = float(point.split('K')[0]) * 1000
        if 'M' in str(point):
            point = float(point.split('M')[0]) * 1000000
        return point

# with open(filename) as infile:
#   data = json.load(infile)



features = []

client = MongoClient()
db = client.influencers
count = 0
collection = db.qoruz

for element in collection.find():
    if str(element['demographics']['success'])=='true':

        print 'Processing Influencer No:{}'.format(count)
        temp_feature = np.asarray([])
        # temp_feature.append(element['major']['info']['name'])
        try:
            temp_feature = np.hstack([temp_feature,model.docvecs[str(element['demographics']['demographic']['twitter']['id'])]])
            temp_feature = np.hstack([temp_feature,replace_none(user.utc_offset)])
        except Exception as error:
            print error
            continue
        # temp_feature.append(replace_none(element['major']['blogger']['domain_authority']))
        # temp_feature.append(replace_none(element['major']['blogger']['reach']))
        # temp_feature.append(replace_none(element['major']['blogger']['score']))



        # temp_feature.append(replace_none(element['major']['facebook']['interaction']))
        # temp_feature.append(replace_none(element['major']['facebook']['likes']))
        # temp_feature.append(replace_none(element['major']['facebook']['talking']))
        # temp_feature.append(replace_none(element['major']['facebook']['score']))


        # temp_feature.append(replace_none(element['major']['instagram']['followers']))
        # temp_feature.append(replace_none(element['major']['instagram']['interaction']))
        # temp_feature.append(replace_none(element['major']['instagram']['media']))
        # temp_feature.append(replace_none(element['major']['instagram']['score']))

        temp_feature = np.hstack([temp_feature,replace_none(element['major']['twitter']['followers'])])
        temp_feature = np.hstack([temp_feature,replace_none(element['major']['twitter']['interaction'])])
        temp_feature = np.hstack([temp_feature,replace_none(element['major']['twitter']['tweets'])])
        temp_feature = np.hstack([temp_feature,replace_none(element['major']['twitter']['score'])])


        # temp_feature.append(replace_none(element['major']['youtube']['subscribers']))
        # temp_feature.append(replace_none(element['major']['youtube']['videos']))
        # temp_feature.append(replace_none(element['major']['youtube']['views']))
        # temp_feature.append(replace_none(element['major']['youtube']['score']))

        # try:
        #   temp_feature.append(replace_none(element['demographics']['demographic']['real_humans']))
        # except:
        #   temp_feature.append(0)
        # try:
        #   temp_feature.append(replace_none(element['demographics']['demographic']['buy_luxary']))
        # except:
        #   temp_feature.append(0)
        
        try:    
            location = geolocs_dict[element['demographics']['demographic']['cities'][0]['name']]

            temp_feature = np.hstack([temp_feature,replace_none(location[0])])
            temp_feature = np.hstack([temp_feature,replace_none(location[1])])
        except Exception as error:
            print error
            continue
        ind_percent = 0
        eng_percent = 0

        try:
            for item in element['demographics']['demographic']['countries']:
                if item['name']=='India':
                    ind_percent = item['percent']
        except Exception as error:
            print error
            continue

        temp_feature = np.hstack([temp_feature,replace_none(ind_percent)])
        try:
            temp_feature = np.hstack([temp_feature,replace_none(element['demographics']['demographic']['gender']['male'])])
        except:
            continue
        features.append(temp_feature)
        count +=1
    else:
        pass


features = np.asarray(features, dtype='float')
X = features[:,:25]
Y = features[:,27]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33)
