from twitter import Api
import yaml
import os
import sys


def _get_api():    
    config = yaml.safe_load(open(os.path.join(sys.path[0], "config.yml")))

    # TODO - check validity of the config obj

    return Api(consumer_key = config['app_key'],
    consumer_secret = config['app_secret'], 
    access_token_key = config['oauth_token'], 
    access_token_secret = config['oauth_token_secret'])
    #print (api.VerifyCredentials())

def _post_misc_msg():
    api = _get_api()
    status = api.PostUpdate('testing from twitter_utils.py')

def _post_misc_img():
    api = _get_api()
    print("getapi done")
    status = api.PostUpdate('testing photo post from twitter_utils.py', media = "bounded.jpg")

def _post_update(num_people, image_raw, image_bounded):
    print ('uploading to twitter...')
    #print ('image_raw-{:s}, bounded-{:s}'.format(image_raw, image_bounded))
    api = _get_api()
    
    # TODO - upload multiple photos
     
    if(num_people == 0):
        tweet_body = 'nobody taking selfies in #DUMBO right now :('
    elif(num_people == 1):
        tweet_body = '1 person taking a selfie in #DUMBO right now'
    else:
        tweet_body = ( str(num_people) + ' people taking selfies in #DUMBO right now' )

    status = api.PostUpdate(tweet_body, media = image_bounded, latitude='40.7034973', longitude='-73.9898414')
    
    # TODO - check status of upload
    
    print ('upload complete!')

def main():
    
    # main() is not intended to run directly, it's used to unit-test uploading to Twitter
    _post_misc_msg()
  
  
if __name__== "__main__":

    main()
