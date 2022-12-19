import aiohttp
from aiohttp import web
import pandas as pd
from pandas import DataFrame
import numpy as np
import re, os, string
from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
import tensorflow as tf
from gensim.models.fasttext import FastText 
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import *
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
PAGE_ACCESS_TOKEN = ''
# verify token
VERIFY_TOKEN = ''
fast_text_model= KeyedVectors.load('./New/new_model.model')
model=tf.keras.models.load_model('./PTHTTM/savemodel/model1.h5');
listmodel=[]
for i in range(1,26):
    listmodel.append(tf.keras.models.load_model('./PTHTTM/savemodel/label'+str(i)+'.h5'))
class BotControl(web.View):
    async def get(self):
        query = self.request.rel_url.query
        if(query.get('hub.mode') == "subscribe" and query.get("hub.challenge")):
            if not query.get("hub.verify_token") == VERIFY_TOKEN:
                return web.Response(text='Verification token mismatch', status=403)
            return web.Response(text=query.get("hub.challenge"))
        return web.Response(text='Forbidden', status=403)

    async def post(self):
        data = await self.request.json()
        if data.get("object") == "page":
            await self.send_greeting("Chào bạn!")
            for entry in data.get("entry"):
                for messaging_event in entry.get("messaging"):
                    if messaging_event.get("message"):
                        sender_id = messaging_event["sender"]["id"]
                        await self.send_message(sender_id,getAnswer(messaging_event["message"]["text"]))
        return web.Response(text='ok', status=200)

    async def send_greeting(self, message_text):
        params = {
            "access_token": PAGE_ACCESS_TOKEN
        }
        headers = {
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "setting_type": "greeting",
            "greeting": {
                "text": message_text
            }
        })
        async with aiohttp.ClientSession() as session:
            await session.post("https://graph.facebook.com/v3.0/me/thread_settings", params=params, headers=headers, data=data)

    async def send_message(self, sender_id, message_text):
        params = {
            "access_token": PAGE_ACCESS_TOKEN
        }
        headers = {
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "recipient": {
                "id": sender_id
            },
            "message": {
                "text": message_text
            }
        })
        async with aiohttp.ClientSession() as session:
            await session.post("https://graph.facebook.com/v3.0/me/messages", params=params, headers=headers, data=data)
    
routes = [
    web.get('/', BotControl, name='verify'),
    web.post('/', BotControl, name='webhook'),
]
max_length_inp = 30
# Loại bỏ các ký tự thừa
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text
def word_segment(sent):
    sent = tokenize(sent)
    return sent
def sentence_embedding(sent):
    content = clean_text(sent) 
    content = word_segment(content)
    inputs = []
    for word in content.split():
      if word in fast_text_model.wv.index_to_key:
        inputs.append(fast_text_model.wv.get_vector(word))
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=max_length_inp,
                                                                dtype='float32',
                                                                padding='post')
    return inputs
def loadData(name):
  data_path= './PTHTTM/Dataset/answer/'+name
  data_content= pd.read_excel(data_path)
  tmp = DataFrame(data_content, columns = ['STT','Name','Content'])
  return tmp
def getLabel(model,question):
    tam= model.predict(sentence_embedding(question))
    tam=tam[0]
    max=0
    for i in range(0,tam.size):
        if(tam[i]>tam[max]):
            max=i
    return max
def getAnswer(question):
    print(question)
    labelParent= getLabel(model,question)
    modelChild= listmodel[labelParent]
    tmp=loadData(str(labelParent)+'.xlsx')
    label= getLabel(modelChild,question)
    data= DataFrame(tmp, columns = ['Content'])
    return str(data['Content'][label-1])
app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app)
