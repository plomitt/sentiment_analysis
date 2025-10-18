import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

url = "https://api.x.com/2/tweets/search/recent"

querystring = {"max_results":"1","sort_order":"relevancy","query":"bitcoin -is:retweet lang:en","tweet.fields":["author_id","id","created_at","text","public_metrics"]}

headers = {"Authorization": f"Bearer {bearer_token}"}

response = requests.get(url, headers=headers, params=querystring)

# Save response as JSON file
response_data = response.json()
with open('twitter_response.json', 'w', encoding='utf-8') as f:
    json.dump(response_data, f, indent=4, ensure_ascii=False)

print(response_data)