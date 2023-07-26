from flask import Flask
from flask import Flask, request, jsonify, render_template
import numpy as np 
import tensorflow as tf 
import pandas as pd
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import streamlit as st 
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

import tensorflow as tf
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import TFBertForSequenceClassification
import pandas as pd
import requests
from bs4 import BeautifulSoup

import json
import plotly
import plotly.express as px




model = tf.keras.models.load_model('SentimentAnalysis_Model.h5')

#model.summary()

def prep_data(text):
  #text=text.lower()
  tokens= tokenizer.encode_plus(text,max_length=512,
                                truncation=True, padding='max_length',
                                add_special_tokens=True, return_token_type_ids=True,
                                return_tensors='tf')
  return{
      'input_ids':tf.cast(tokens['input_ids'],tf.int32),
      'attention_mask':tf.cast(tokens['attention_mask'],tf.int32)

  }

headers = {
    'authority': 'www.amazon.in',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    # Requests sorts cookies= alphabetically
    # 'cookie': 'session-id=259-3113978-6678618; i18n-prefs=INR; ubid-acbin=260-8554202-6973909; lc-acbin=en_IN; csm-hit=tb:BS866TA0AKH6X86N924E+sa-7XYTQAXQHJP5ADH88228-DY27HYE0CK5V9FW24GBD|1656009294944&t:1656009294945&adb:adblk_yes; session-token=Z1j175VoYxPr2Un/9ciL3Q6lKw+QtLYYIwSQ+GLxjT06952u8vOZromD4WcFE0bs+yrUyLPy8HmIn7mTjUt8qsx3n0meC7yWKFqqwDEm5iecYedklsrNwmDrQOiaMH9lpacbdB8kgUk5IbZdg1VyhrdnY4OZrk6r350ARDEXJExuu2GZr0sV4fpbwUes/V9fDrfASeMQhVEEzmEAAHWN2g==; session-id-time=2082758401l',
    'device-memory': '8',
    'downlink': '10',
    'dpr': '0.8',
    'ect': '4g',
    #'referer': 'https://www.amazon.in/Dell-Vostro-Laptop-i3-1215U-35-56Cms/product-reviews/B0BQJ68HHC/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1',
    'rtt': '0',
    'sec-ch-device-memory': '8',
    'sec-ch-dpr': '0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-viewport-width': '2400',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'service-worker-navigation-preload': 'true',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
    'viewport-width': '2400',
}

def create_headers(url):
  headers['referer']=url
  return headers

def create_url(asin):
  url=f'https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1'
  return url

def create_urls(asin):
  urls=[]
  for i in range(1,4):
    url=f'https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber={i}'
    urls.append(url)
  return urls

def get_soup(url):
    r = requests.get(url, headers=headers,
    params={'url': url, 'wait': 2})
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

reviewlist=[]

def get_reviews(soup):
        # Function 2: look for web-tags in our soup, then append our data to reviewList
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
                #'Rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'Title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            'Review': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            'Review_Date': item.find('span', {'data-hook': 'review-date'}).text.replace('Reviewed in India 🇮🇳 on', '').strip()
            }
            reviewlist.append(review)
    except:
        pass



y=[]
z=[]

def get_avg_sentiment(positive_sentiment,negative_sentiment):
  Avg_positive_sentiment=sum(positive_sentiment)/len(positive_sentiment)
  Avg_negative_sentiment=sum(negative_sentiment)/len(negative_sentiment)
  print(Avg_positive_sentiment)
  print(Avg_negative_sentiment)
  y.append(Avg_positive_sentiment)
  z.append(Avg_negative_sentiment)
  return Avg_positive_sentiment,Avg_negative_sentiment

def sentimental_analysis(reviews):
  sen_analysis=[]
  for i in reviews:
    test=prep_data(i)
    probs=model.predict(test,verbose=0)
    sen_analysis.append(probs)
  #print(sen_analysis)
  positive_sentiment=[]
  negative_sentiment=[]
  for i in sen_analysis:
    positive_sentiment.append(i[0][1])
    negative_sentiment.append(i[0][0])
  #print(positive_sentiment)
  #print(negative_sentiment)
  Avg_positive_sentiment = sum(positive_sentiment) / len(positive_sentiment)
  Avg_negative_sentiment = sum(negative_sentiment) / len(negative_sentiment)
  print(Avg_positive_sentiment)
  print(Avg_negative_sentiment)
  y.append(Avg_positive_sentiment)
  z.append(Avg_negative_sentiment)
  return Avg_positive_sentiment, Avg_negative_sentiment
  #get_avg_sentiment(positive_sentiment,negative_sentiment)

#sentimental_analysis(df['Review'])


#selected_product='HP Victus Gaming Latest'
def create_review_df(prod):
    asin=products.loc[products['Model'] == prod, 'ASIN'].iloc[0]
    url=create_url(asin)
    #reviewlist=[]
    for x in range(1,4):
        #rl1=[]
        #soup = get_soup(f'https://www.amazon.in/product-reviews/B0BQJ68HHC/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber={x}')
        soup=get_soup(url[:-1]+str(x))
        #print(f'Getting page: {x}')
        get_reviews(soup)
        print(len(reviewlist))
        #r=get_reviewlist(soup)
        #rl1.append(r)"""

    df = pd.DataFrame(reviewlist)
    return df


model = tf.keras.models.load_model('SentimentAnalysis_Model.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
products=pd.read_csv("products.csv")
print(products.head())



def sentimental_analysis(reviews):
  sen_analysis=[]
  for i in reviews:
    test=prep_data(i)
    probs=model.predict(test)
    sen_analysis.append(probs)
  print(sen_analysis)
  positive_sentiment=[]
  negative_sentiment=[]
  for i in sen_analysis:
    positive_sentiment.append(i[0][1])
    negative_sentiment.append(i[0][0])
  print(positive_sentiment)
  print(negative_sentiment)
  #get_avg_sentiment(positive_sentiment,negative_sentiment)
  #return Avg_positive_sentiment,Avg_negative_sentiment
  Avg_positive_sentiment=sum(positive_sentiment)/len(positive_sentiment)
  Avg_negative_sentiment=sum(negative_sentiment)/len(negative_sentiment)
  y=[]
  z=[]
  print(Avg_positive_sentiment)
  print(Avg_negative_sentiment)
  y.append(Avg_positive_sentiment)
  z.append(Avg_negative_sentiment)
  return Avg_positive_sentiment,Avg_negative_sentiment

def get_avg_sentiment(positive_sentiment,negative_sentiment):
  Avg_positive_sentiment=sum(positive_sentiment)/len(positive_sentiment)
  Avg_negative_sentiment=sum(negative_sentiment)/len(negative_sentiment)
  print(Avg_positive_sentiment)
  print(Avg_negative_sentiment)
  y.append(Avg_positive_sentiment)
  z.append(Avg_negative_sentiment)
  




'''fig, ax = plt.subplots()

# Plot the lines
ax.plot(x, a, marker='o', label='Positive sentiment')
#ax.plot(x, z, marker='o', label='Negative sentiment')

ax.set_xticklabels(x, rotation=45)

ax.set_ylabel('Sentiment Value')

ax.set_title('Sentimental trend Analysis of recent 6 months')

ax.legend()

plt.show()'''


app = Flask(__name__,template_folder='template')

def prep_data(text):
  #text=text.lower()
  tokens= tokenizer.encode_plus(text,max_length=512,
                                truncation=True, padding='max_length',
                                add_special_tokens=True, return_token_type_ids=True,
                                return_tensors='tf')
  return{
      'input_ids':tf.cast(tokens['input_ids'],tf.int32),
      'attention_mask':tf.cast(tokens['attention_mask'],tf.int32)
    }

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/bar.html")
def bar():
    return render_template('bar.html')

@app.route("/example.html",methods=['POST','GET'])
def example():
    if request.method == 'POST':
        l1=len(reviewlist)
        print(l1)
        product=request.form.get('search-input')
        revs=create_review_df(product)
        l2=len(reviewlist)
        print(l2)
        while(l1==l2):
            revs=create_review_df(product)
            l2=len(reviewlist)
        else:
            revs1=revs.tail(10)
            print(revs1)
            a,b=sentimental_analysis(revs1['Review'])
            print(a)
            print(b)
        
            l=[['Positive',a],['Negative',b]]
        # Convert list to dataframe and assign column value
            df = pd.DataFrame(l,
                            columns=['Sentiment','Score'])
            
            # Create Bar chart
            #fig = px.bar(df, x='Name', y='Age', color='City', barmode='group')
            #fig = px.bar(df, x='Sentiment', y='Score', color='Sentiment', barmode='group')
            fig = px.bar(df, x='Sentiment', y='Score', color='Sentiment',color_discrete_map={'Positive':'green','Negative':'red'},barmode='group')
            
            # Create graphJSON
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Use render_template to pass graphJSON to html
            return render_template('bar.html', graphJSON=graphJSON,prod_name=product)
    else:
        return render_template('example.html')
    

@app.route("/example1.html",methods=['POST','GET'])
def example1():
    if request.method == 'POST':
        l1=len(reviewlist)
        print(l1)
        product=request.form.get('search-input')
        revs=create_review_df(product)
        l2=len(reviewlist)
        print(l2)
        while(l1==l2):
            revs=create_review_df(product)
            l2=len(reviewlist)
        else:
            revs1=revs.tail(10)
            print(revs1)
        
            l=revs1['Review'].tolist()
            print(l)
        # Convert list to dataframe and assign column value
            def make_sent(l):
              all=[]
              for i in l:
                  text=i
                  descr=[]
                  senti=[]
                  sentences=nltk.tokenize.sent_tokenize(text)
                  for j in sentences:
                      tf_batch=prep_data(j)
                      tf_outputs=model.predict(tf_batch)
                      labels=['Negative','Positive']
                      label=tf.argmax(tf_outputs,axis=1)
                      label=label.numpy()
                      #print(j,": \n")
                      descr.append(j)
                  all.append(descr)

              descr_df=pd.DataFrame(all)
              return descr_df
            make_sent(l)
            from pyabsa import ATEPCCheckpointManager
            aspect_extractor= ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',auto_device=True)
            inference_source=l
            atepc_results=aspect_extractor.extract_aspect(inference_source=inference_source,pred_sentiment=True,)
            print(atepc_results[0]['aspect'])
            print(atepc_results[0]['sentiment'])
            print(atepc_results[1]['aspect'])
            print(atepc_results[1]['sentiment'])
            ta={}
            for i in range(len(l)):
                for j in range(len(atepc_results[i]['aspect'])):
                    ta[atepc_results[i]['aspect'][j]]=atepc_results[i]['sentiment'][j]
            print(ta)
            negative_asp=[]
            positive_asp=[]
            for i in ta:
                if ta[i]=='Negative':
                    negative_asp.append(i)
                else:
                    positive_asp.append(i)
            print(positive_asp)
            print(negative_asp)
            absa=pd.read_json("atepc_inference.result.json")
            absa1=absa[['aspect','sentiment','confidence']]
            absa_dict=absa1.to_dict()
            asps=list(absa_dict['aspect'].values())
            sents=list(absa_dict['sentiment'].values())
            conf=list(absa_dict['confidence'].values())
            asps_new= list(np.concatenate(asps))
            sents_new= list(np.concatenate(sents))
            conf_new= list(np.concatenate(conf))
            for i in range(len(asps_new)):
                if sents_new[i]=='Negative':
                    conf_new[i]=conf_new[i]*(-1)
            df = pd.DataFrame(list(zip(asps_new,conf_new)),
               columns =['Aspect', 'Score'])
            df1=df.groupby(['Aspect'])
            print(df1)

            return render_template('result1.html', positive_aspects=positive_asp, negative_aspects=negative_asp,prod_name=product)
    return render_template('example1.html')



    

@app.route("/index1.html")
def index1():
    return render_template('index1.html')

@app.route("/index.html")
def index():
    return render_template('index.html')

@app.route("/signup.html")
def signup():
    return render_template('signup.html')
@app.route("/signin.html")
def signin():
    return render_template('signin.html')
@app.route("/profile.html")
def profile():
    return render_template('profile.html')

if __name__ == "__main__":
    app.run(debug=True)