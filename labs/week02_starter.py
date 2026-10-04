"""Week 2 student starter: factorial feature/model comparison on synthetic text."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def sigmoid(z):
    z=np.asarray(z); return np.exp(-np.logaddexp(0,-z))

def loss_gradient(X,y,w,b,l2=0.):
    """Mean binary NLL + l2/2 times squared weight norm."""
    raise NotImplementedError("Complete this laboratory task.")

def train_logistic(X,y,steps=800,lr=.5,l2=.001):
    raise NotImplementedError("Complete this laboratory task.")

def data():
    texts=[]; labels=[]
    for subject in ['sensor','radio','camera','screen','battery','motor','cable','speaker']:
        for adjective,positive in [('good',1),('reliable',1),('poor',0),('faulty',0)]:
            for negated in [False,True]:
                for place in ['today','indoors','outside','again','overnight']:
                    texts.append(f'the {subject} was {"not " if negated else ""}{adjective} {place}')
                    labels.append(int(bool(positive)!=negated))
    ids=np.arange(len(texts)); y=np.array(labels)
    tr,other=train_test_split(ids,test_size=.4,stratify=y,random_state=17)
    dev,te=train_test_split(other,test_size=.5,stratify=y[other],random_state=18)
    return np.array(texts),y,tr,dev,te

def cosine(a,b):
    """Return cosine similarity; define it as 0 if either vector has zero norm."""
    raise NotImplementedError("Complete this laboratory task.")

def paired_bootstrap(y,a,b,iterations=500,seed=17):
    """CI for accuracy(a)-accuracy(b), paired by document."""
    raise NotImplementedError("Complete this laboratory task.")

def checks():
    X=np.array([[2.,-1.],[0.,1.]]); y=np.array([1.,0.]); w=np.array([.3,-.2]); b=.1
    _,g,_=loss_gradient(X,y,w,b)
    for i in range(2):
        d=np.zeros(2); d[i]=1e-5
        numeric=(loss_gradient(X,y,w+d,b)[0]-loss_gradient(X,y,w-d,b)[0])/2e-5
        assert np.isclose(g[i],numeric,atol=1e-6)
    assert np.isclose(cosine(np.array([1,1,0]),np.array([1,0,1])),.5)

def experiment():
    texts,y,tr,dev,te=data()
    vectorizer=TfidfVectorizer(); raw=vectorizer.fit_transform(texts[tr])
    X={name:vectorizer.transform(texts[ix]).toarray() for name,ix in [('train',tr),('dev',dev),('test',te)]}
    svd=TruncatedSVD(n_components=16,random_state=17).fit(raw)
    dense={name:svd.transform(value) for name,value in X.items()}
    results={}; predictions={}
    for feature,views in [('tfidf',X),('svd16',dense)]:
        for model in ['linear','mlp']:
            if model=='linear':
                w,b=train_logistic(views['train'],y[tr]); predict=lambda z:(sigmoid(z@w+b)>=.5).astype(int)
            else:
                fit=MLPClassifier(hidden_layer_sizes=(24,),max_iter=1000,random_state=17,alpha=.001,learning_rate_init=.01).fit(views['train'],y[tr]); predict=fit.predict
            a=predict(views['dev']); p=predict(views['test']); key=feature+'_'+model
            results[key]={'dev_macro_f1':f1_score(y[dev],a,average='macro'),'test_macro_f1':f1_score(y[te],p,average='macro'),'test_accuracy':accuracy_score(y[te],p),'confusion':confusion_matrix(y[te],p).tolist()}
            predictions[key]=p
    results['paired_accuracy_ci_tfidf_mlp_minus_linear']=paired_bootstrap(y[te],predictions['tfidf_mlp'],predictions['tfidf_linear'])
    results['note']='SVD produces corpus-derived dense vectors; these are not pretrained word2vec embeddings.'
    return results

if __name__=='__main__':
    checks(); print(experiment())
