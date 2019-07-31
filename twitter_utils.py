from twitter import Api
import yaml


def _get_api():
    return Api(consumer_key = config['APP_KEY'],
    consumer_secret = config['APP_SECRET'], 
    access_token_key = ['OAUTH_TOKEN'], 
    access_token_secret = ['OAUTH_TOKEN_SECRET')
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
   
    print("about to post update")
    status = api.PostUpdate(tweet_body, media = image_bounded, latitude='40.7034973', longitude='-73.9898414')
    
    # TODO - check status of upload
    
    print ('upload complete!')


def main():
    # postRandomMsg()
    #postRandomImage()
    _post_misc_msg()
  
  
if __name__== "__main__":
  main()
  config = yaml.safe_load(open(os.path.join(sys.path[0], "config.yml")))
