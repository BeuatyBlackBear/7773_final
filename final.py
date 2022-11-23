from metaflow import FlowSpec, step, Parameter, IncludeFile, current
import requests
from datetime import datetime
import os

import pandas as pd
import seaborn as sns
import matplotlib 

pd.set_option('display.max_columns',None)
pd.options.display.max_seq_items = 2000
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import requests, re
import nltk
import string, itertools
from collections import Counter, defaultdict
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.cluster import KMeans
from wordcloud import WordCloud

import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'




class MyClassificationFlow(FlowSpec):
    TEST_SPLIT = Parameter(name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20)

    ## get dataset by category
    def get_dataset(self,restaurants_reviews,category):
        df = restaurants_reviews[['removed_punct_text','labels']][restaurants_reviews.category==category]
        df.reset_index(drop=True, inplace =True)
        df.rename(columns={'removed_punct_text':'text'}, inplace=True)
        return df
    ## only keep positive and negative words
    def filter_words(self,review):
            words = [word for word in review.split() if word in positive_words + negative_words]
            words = ' '.join(words)
            return words
 
    ## 得到重要性评分
    def get_polarity_score(self,dataset,positive_words,negative_words):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.svm import LinearSVC

        dataset.text = dataset.text.apply(self.filter_words)
        
        terms_train=list(dataset['text'])
        class_train=list(dataset['labels'])
        
        ## get bag of words
        vectorizer = CountVectorizer()
        feature_train_counts=vectorizer.fit_transform(terms_train)
        
        ## run model
        svm = LinearSVC()
        svm.fit(feature_train_counts, class_train)
        
        ## create dataframe for score of each word in a review calculated by svm model
        coeff = svm.coef_[0]
        cuisine_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
        
        ## get frequency of each word in all reviews in specific category
        cuisine_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
        cuisine_reviews['labels'] = class_train
        cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
        
        cuisine_words_score.set_index('word', inplace=True)
        cuisine_polarity_score = cuisine_words_score
        cuisine_polarity_score['frequency'] = cuisine_frequency
        
        cuisine_polarity_score.score = cuisine_polarity_score.score.astype(float)
        cuisine_polarity_score.frequency = cuisine_polarity_score.frequency.astype(int)
        
        ## calculate polarity score 
        cuisine_polarity_score['polarity'] = cuisine_polarity_score.score * cuisine_polarity_score.frequency / cuisine_reviews.shape[0]
        
        cuisine_polarity_score.polarity = cuisine_polarity_score.polarity.astype(float)
        ## drop unnecessary words
        unuseful_positive_words = ['great','amazing','love','best','awesome','excellent','good',
                                                       'favorite','loved','perfect','gem','perfectly','wonderful',
                                                        'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                        'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                        'fabulous','liked','incredible','outstanding','positive']
        unuseful_negative_words =  ['bad','disappointed','disappointing','horrible','disappoint','lacking','unfortunately','sorry']
        unuseful_words = unuseful_positive_words + unuseful_negative_words
        cuisine_polarity_score.drop(cuisine_polarity_score.loc[unuseful_words].index, axis=0, inplace=True)
        
        return cuisine_polarity_score,vectorizer,svm

    def get_top_words(self,dataset, label, number=20):
        if label == 'positive':
            df = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
        else:
            df = dataset[dataset.polarity<0].sort_values('polarity')[:number]
        return df
    def split_data(self,dataset, test_size):
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
        return df_train,df_test

    def test_data(self,dataset,transform,model):

        x_test=list(dataset['text'])
        y_test=list(dataset['labels'])
        
        ## get bag of words
    
        feature_train_counts=transform.transform(x_test)
        y_predict=model.predict(feature_train_counts)
        
        from sklearn.metrics import accurancy_score
        score=accurancy_score(y_predict,y_test)
        
        return score
    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_business_data,self.load_review_data)
    @step
    def load_business_data(self):
        import json
        import pandas as pd
        data_file_1 = open("./archive/yelp_academic_dataset_business.json")
        data_1 = []
        for line in data_file_1:
            data_1.append(json.loads(line))
        self.business = pd.DataFrame(data_1)
        data_file_1.close()

        self.next(self.business_data_preprocessing)
    @step
    def business_data_preprocessing(self):
        ## drop unuseful column 'hours','attributes'
        business=self.business
        business.drop(['hours','attributes'], axis=1, inplace=True)

        ## remove quotation marks in name and address column
        business.name=business.name.str.replace('"','')
        business.address=business.address.str.replace('"','')#把引号去掉

        ## 按洲名简写筛选数据，
        ## 并存入新的dataframe：use
        states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        usa=business.loc[business['state'].isin(states)] #把不是美国州的去掉
        usa=usa.dropna(axis=0, subset=['categories'])

        ## 选出所有餐厅，
        ## 并存入新的dataframe：us_restaurants
        us_restaurants=usa[usa['categories'].str.contains('Restaurants')]

         ## select out 16 cuisine types of restaurants and rename the category
         ## 把 us_restaurants['category'] 中含有指定种类的餐厅筛选出来，
          ## 并新建column：'category'

         # us_restaurants.is_copy=False 版本过老
        us_restaurants['category']=pd.Series()
        us_restaurants.loc[us_restaurants.categories.str.contains('American'),'category'] = 'American'
        us_restaurants.loc[us_restaurants.categories.str.contains('Mexican'), 'category'] = 'Mexican'
        us_restaurants.loc[us_restaurants.categories.str.contains('Italian'), 'category'] = 'Italian'
        us_restaurants.loc[us_restaurants.categories.str.contains('Japanese'), 'category'] = 'Japanese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Chinese'), 'category'] = 'Chinese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Thai'), 'category'] = 'Thai'
        us_restaurants.loc[us_restaurants.categories.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
        us_restaurants.loc[us_restaurants.categories.str.contains('French'), 'category'] = 'French'
        us_restaurants.loc[us_restaurants.categories.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Greek'),'category'] = 'Greek'
        us_restaurants.loc[us_restaurants.categories.str.contains('Indian'),'category'] = 'Indian'
        us_restaurants.loc[us_restaurants.categories.str.contains('Korean'),'category'] = 'Korean'
        us_restaurants.loc[us_restaurants.categories.str.contains('Hawaiian'),'category'] = 'Hawaiian'
        us_restaurants.loc[us_restaurants.categories.str.contains('African'),'category'] = 'African'
        us_restaurants.loc[us_restaurants.categories.str.contains('Spanish'),'category'] = 'Spanish'
        us_restaurants.loc[us_restaurants.categories.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
        us_restaurants[:20]

        
        ## drop null values in category, 
        us_restaurants=us_restaurants.dropna(axis=0, subset=['category'])

        del us_restaurants['categories']

        ## and reset the index
        self.us_restaurants=us_restaurants.reset_index(drop=True)
        
        self.next(self.join_two_dataset)

    @step
    def load_review_data(self):
        ## load review table
        # review = pd.read_csv('yelp_review.csv')
        # review.head()
        import json
        import pandas as pd
        data_file_2 = open("./archive/yelp_academic_dataset_review.json")
        data_2 = []

        for line in data_file_2:
            data_2.append(json.loads(line))
        self.review = pd.DataFrame(data_2)
        data_file_2.close()

        self.next(self.join_two_dataset)

       
    @step
    def join_two_dataset(self,inputs):
        ## 以 'business_id' 为准合并两个df，得到 “restaurants_reviews”
        self.restaurants_reviews = pd.merge(inputs.business_data_preprocessing.us_restaurants, inputs.load_review_data.review, on = 'business_id')
        
        self.next(self.generate_labels_and_preprocessing)

    @step
    def generate_labels_and_preprocessing(self):

        restaurants_reviews=self.restaurants_reviews
        ## 更新 column names
        restaurants_reviews.rename(columns={'stars_x':'avg_star','stars_y':'review_star'}, inplace=True)

        ## 把评论中所有的：特殊符号和换行符 全部用空格替换，然后计算每个评论的字数并存入 'num_words_review'
        restaurants_reviews['num_words_review'] = restaurants_reviews.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))


        # 根据星级，对评论 label reviews as positive or negative
        restaurants_reviews['labels'] = ''
        restaurants_reviews.loc[restaurants_reviews.review_star >=4, 'labels'] = 'positive'
        restaurants_reviews.loc[restaurants_reviews.review_star ==3, 'labels'] = 'neural'
        restaurants_reviews.loc[restaurants_reviews.review_star <3, 'labels'] = 'negative'

        # drop neutral reviews for easy analysis
        restaurants_reviews.drop(restaurants_reviews[restaurants_reviews['labels'] =='neural'].index, axis=0, inplace=True)
        restaurants_reviews.reset_index(drop=True, inplace=True)


    
        pd.set_option('display.float_format', lambda x: '%.4f' % x)


        ## convert text to lower case 全部转化为小写
        restaurants_reviews.text = restaurants_reviews.text.str.lower()

        ## remove unnecessary punctuation（又重复操作了一遍）
        restaurants_reviews['removed_punct_text']= restaurants_reviews.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')

        # ## 将 positive file 导入 list：positive_words中
        # file_positive = open('/Users/houdeliao/Desktop/positive.txt')
        # reader =csv.reader(file_positive)
        # positive_words = [word[0] for word in reader]
        
        self.restaurants_reviews=restaurants_reviews
        self.next(self.load_pos_neg_words)

    

    @step
    def load_pos_neg_words(self):

        import sys
        positive_words=[]
        with open('./positive.txt','r') as p:
            for pline in p:
                positive_words.append(pline.strip('\n'))

        self.positive_words=positive_words

        ## 将 negative file 导入 list：negative_words中
        file_negative = open('./negative_.txt')
        reader =csv.reader(file_negative)
        self.negative_words = [word[0] for word in reader]
        self.next(self.japanese_example)



        # negative_words=[]
        # with open('/Users/houdeliao/Desktop/negative.txt','r') as n:
        #     for nline in n:
        #         negative_words.append(nline.strip('\n'))
        # print(negative_words)

        
    @step
    def japanese_example(self):
        restaurants_reviews=self.restaurants_reviews
        positive_words=self.positive_words
        negative_words=self.negative_words
        Japanese_reviews = self.get_dataset(restaurants_reviews,'Japanese')
        Japanese_train,Japanese_test = self.split_data(Japanese_reviews, 0.9)
        print('Total %d number of reviews' % Japanese_train.shape[0])

        Japanese_polarity_score,vectorizer_Jap,svm_Jap = self.get_polarity_score(Japanese_train,positive_words,negative_words)
        print(self.get_top_words(Japanese_polarity_score, 'positive',20))

        print(self.get_top_words(Japanese_polarity_score,'negative',20))

        Japanese_test_score=self.test_data(Japanese_test,vectorizer_Jap,svm_Jap)
        print("The accurancy_score on Japanese test set is:{}".format(Japanese_test_score))
        self.vectorizer_Jap=vectorizer_Jap
        self.svm_Jap=svm_Jap
        self.next(self.end)

    
    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))



if __name__ == '__main__':
    MyClassificationFlow()















    