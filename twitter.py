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

def get_demographics(handle):
	from persephone.classify import classified
	os.chdir('persephone')
	demographics = classified(handle)
	return demographics 

#Request for user, if found proceeed, else quit#
handle = raw_input('Query handle? \t')
try:
	user = api.get_user(handle)
except:
	print 'GG. Quiting.'
	import sys
	sys.exit(0)
print 'Found User: ' + str(user.screen_name) + ' with Follower Count:' + str(user.followers_count)


#Get Basic Details from Query
print 'Extracting Basic Data:'
followers_count = user.followers_count
favourites_count = user.favourites_count
friends_count = user.friends_count
description = user.description
language = user.lang
location = user.location
name = user.name
tweets_count = user.statuses_count
verified = user.verified

#Get advanced stats using recent tweets
recent_tweets = user.timeline(count=200)
analysis_texts = []
for status in user.timeline(count=200):
    analysis_texts.append(status.text)
# sentiments = get_sentiment_analysis(analysis_texts)
# topic_vector = get_topic_vector(analysis_texts)
demographics = get_demographics(handle)
print 'Demographics:'
print(demographics)