import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open('intents.json') as file:
    data = json.load(file)
try:
    with open('data.pickle','rb') as f :
        wrds,labls,train,output = pickle.load(f)

except:


    wrds=[]
    labls=[]
    pat1=[]
    pat2=[]
    for i in data['intents'] :
        for p in i['patterns']:
            wrd=nltk.word_tokenize(p)
            pat1.append(wrd) 
            pat2.append(i['tag'])
            
            wrds.extend(wrd)
        if i['tag'] not in labls :
            labls.append(i['tag'])
    wrds=[stemmer.stem(w.lower() )for w in wrds if w != '?']
    wrds=sorted(list(set(wrds)))
    labls=sorted(labls)
    train=[]
    output=[]
    out_emp=[0 for _ in range(len(labls))]
    for x, doc in enumerate(pat1):
        bag=[]
        words = [stemmer.stem(w) for w in doc ]
        for w in wrds :
            if w in words :
                bag.append(1)
            else:
                bag.append(0)
        out=out_emp[:]
        out[labls.index(pat2[x])]=1 
        train.append(bag)
        output.append(out)
    train=numpy.array(train)
    output = numpy.array(output)
   
    with open('data.pickle','wb') as f :
        pickle.dump((wrds,labls,train,output),f)


tensorflow.compat.v1.reset_default_graph()
net=tflearn.input_data(shape=[None,len(train[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation='softmax')
net=tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load('model.tflearn')
except:
    model.fit(train,output, n_epoch=1000 , batch_size=8 , show_metric=True )
    model.save('model.tflearn')

def bag_words(s,wrds):
    bag=[0 for _ in range (len(wrds))]
    s_wrds=nltk.word_tokenize(s)
    s_wrds=[stemmer.stem(wrd.lower()) for wrd in s_wrds]
    for s in s_wrds:
        for i,w in enumerate(wrds):
            if w== s :
               bag[i] = 1
    return numpy.array(bag)
def chat():
    print('START CHATTING WITH THE BOT! (TYPE "quit" TO STOP )!')
    while True:
        inp = input('YOU: ').strip().lower()

        if inp in ["quit", "exit", "stop"]:
            print("Bot: Goodbye 👋")
            break

        pred = model.predict([bag_words(inp, wrds)])
        res_index = numpy.argmax(pred)
        tag = labls[res_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
                break

        print(random.choice(responses))
chat()






