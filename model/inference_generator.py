import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

import nltk
from nltk import word_tokenize, sent_tokenize
from collections import Counter, defaultdict

import string

from nltk.corpus import wordnet
import spacy
import torch
import json 
from collections import Counter
import itertools
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import networkx as nx
from sklearn.decomposition import PCA
import spacy
from spacy.symbols import nsubj, VERB, NOUN, PROPN, ADV, ADJ

class  inference_gen:

    def __init__(self):
      self.context_vectors = []
      self.target_words = []
      self.candidate_word_set = []
      self.model = gensim.models.Word2Vec.load("model\word2vec_v3.model")
      self.metaphor = []
      self.sentences_tokenized = []
      self.summary = ""
      self.gen_text = []
      self.abstractive_summary = ""
      self.replace_word=[]
      self.target_vector = []
      self.rpl_vectors = []
      self.word_embeddings = []
      self.scores = []
      self.pca = ""
      self.ca_scores =[]
      self.target_word_pairs = []
      self.f_con_vec =[]
      self.pca_variance = []

      
    def cos_sim(self,a, b):
      x = -1 
      y = 1
      dp = np.dot(a, b)
      norm_a = np.linalg.norm(a)
      norm_b = np.linalg.norm(b)
      val = dp / (norm_a * norm_b)
      return (val-x)/(y-x)

    
    def pcaf(self):
      self.pca = PCA(n_components=3)
      self.pca.fit(self.model[self.model.wv.vocab])
      self.pca_variance = self.pca.explained_variance_ratio_

    def map_word_frequency(self,document):
      return Counter(itertools.chain(*document))

    # reference: https://intellica-ai.medium.com/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c
    def get_sif_feature_vectors(self,sentence,sentences):
        sentence1 = [token for token in sentences if token in list(self.model.wv.vocab.keys())]
        word_counts = self.map_word_frequency((sentence1))
        embedding_size = 100 # size of vectore in word embeddings
        a = 0.001
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
          if word in self.model.wv.vocab:
              a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
              vs = np.add(vs, np.multiply(a_value, self.model.wv[word])) # vs += sif * word_vector
        vs = np.divide(vs, sentence_length) # weighted average
        return vs
    def abstract_score_preprocessing(self):
      f = open("model\mrc_data.txt", "r")
      data = f.read().split('\n')
      for row in data:
          word = row.strip()[:-3].strip().lower()
          if(word in self.model.wv.vocab):
              self.word_embeddings.append([self.model.wv[word]])
              self.scores.append(int(row.strip()[-3:]))

    def abstract_score_pred(self, target_word):
      target_word = str(target_word)
      if target_word in self.model.wv.vocab:
        test_embeddings = [self.model.wv[target_word]]
        max_sim = 0
        max_sim_word = ''

        for i in range(len(self.word_embeddings)):
            
            if(self.cos_sim(test_embeddings[0],self.word_embeddings[i][0]) > max_sim):
                max_sim = self.cos_sim(test_embeddings[0],self.word_embeddings[i][0])
                max_sim_abstract_score = self.scores[i]
            
        return max_sim_abstract_score
      else:
        return -1

    def find_target_words(self,sentences):

      for sentence in  nltk.sent_tokenize(sentences):
          
          nlp = spacy.load("en_core_web_sm")
          doc = nlp(sentence)

          pairs = set()
          t_dict = {}
          for possible_subject in doc:
              if possible_subject.pos in [NOUN,PROPN] and possible_subject.head.pos == VERB:
                  pairs.add((possible_subject,possible_subject.head))
          t_dict['nv'] = pairs
          pairs = set()
          for possible_subject in doc:
               if possible_subject.head.pos in [NOUN,PROPN] and possible_subject.pos == ADJ:
                  pairs.add((possible_subject.head,possible_subject))
          t_dict['na'] = pairs
          pairs = set()
          for possible_subject in doc:
               if possible_subject.head.pos == VERB and possible_subject.pos == ADV:
                  pairs.add((possible_subject.head,possible_subject))
          t_dict['va'] = pairs

          max_abs_diff = 0
          target_max_abs = ''

          ca_scores_temp = []

          #nv
          nv_data = t_dict['nv']
          for pair in nv_data:
            score1 = self.abstract_score_pred(pair[0])
            score2 = self.abstract_score_pred(pair[1])
            if(score1 != -1 and score2 != -1):
              curr_abs_diff = abs(score1 - score2)
            else:
              curr_abs_diff = 0

            ca_scores_temp.append((pair,curr_abs_diff))
   
            if(curr_abs_diff > max_abs_diff):
              max_abs_diff = curr_abs_diff
              target_max_abs = pair[1]

          #na
          na_data = t_dict['na']
          for pair in na_data:
            score1 = self.abstract_score_pred(pair[0])
            score2 = self.abstract_score_pred(pair[1])
            if(score1 != -1 and score2 != -1):
              curr_abs_diff = abs(score1 - score2)
            else:
              curr_abs_diff = 0
            
            ca_scores_temp.append((pair,curr_abs_diff))
   
            if(curr_abs_diff > max_abs_diff):
              max_abs_diff = curr_abs_diff
              target_max_abs = pair[1]
          
          #va
          va_data = t_dict['va']
          for pair in va_data:
            score1 = self.abstract_score_pred(pair[0])
            score2 = self.abstract_score_pred(pair[1])
            if(score1 != -1 and score2 != -1):
              curr_abs_diff = abs(score1 - score2)
            else:
              curr_abs_diff = 0
            
            ca_scores_temp.append((pair,curr_abs_diff))
   
            if(curr_abs_diff > max_abs_diff):
              max_abs_diff = curr_abs_diff
              target_max_abs = pair[1]

          self.ca_scores.append(ca_scores_temp)
          sub_toks = [str(target_max_abs)]

          s=[]
          if len(sub_toks)!=0 and sub_toks[0] in self.model.wv.vocab:
            target = str(sub_toks[0])
            for w in nltk.word_tokenize(sentence):
              if(w != target):
                s.append(w)
            self.target_vector.append(self.model.wv[target])
            self.target_words.append(target)
          else:
            s=nltk.word_tokenize(sentence)
            self.target_vector.append("")
            self.target_words.append("")

          self.sentences_tokenized.append(s)


    def compute_context_vector(self,context):
      context_vector = []
      context_vector.append(self.get_sif_feature_vectors(context,self.sentences_tokenized))
      return context_vector

    def find_context_sims(self,context):
      sims = []
      for cv in self.context_vectors:
        sims.append(self.cos_sim(context[0],cv[0]))
         
      prop = int(0.3*len(self.sentences_tokenized))
      pop = sorted(range(len(sims)), key=lambda x: sims[x])[-prop:] # prop -> number of sentences to consider
     
      pop.reverse()

      final_context_vector = [0 for i in range(len(context[0]))]
      for node in pop:
          # print()
          temp = [cv for cv in self.context_vectors[node][0]]
          final_context_vector = [c1 + c2 for c1,c2 in zip(temp,final_context_vector)]
      final_context_vector[0] = np.array(final_context_vector[0])/prop
      return final_context_vector,pop

    def find_candidate_wordset(self,t_word):
      synonyms = []
      hypernyms = []
      #print("Traget word selected: "+t_word)
      if t_word!="":
        for syn in wordnet.synsets(t_word):
              for hyper in syn.hyponyms():
                      hypernyms.append(hyper.lemma_names()[0])
              for l in syn.lemmas():
                  synonyms.append(l.name())

        self.candidate_word_set = set(synonyms+hypernyms)

      #print(self.candidate_word_set)

    def find_cos_sim(self,context_vector):
        cos_sim = []
        
        if len(self.candidate_word_set):
          for k in self.candidate_word_set:
              if self.model.wv.vocab.get(k):
                  cos_sim.append(self.cos_sim(context_vector,self.model.wv.get_vector(k))[0])
              else:
                  cos_sim.append(0)
        return cos_sim
    def predict_metaphor(self,cos_sim,targ_wrd,txt):
     
      if self.model.wv.vocab.get(targ_wrd) and len(list(self.candidate_word_set))>0 and list(self.candidate_word_set)[cos_sim.index(max(cos_sim))] in self.model.wv.vocab:
            rpw = list(self.candidate_word_set)[cos_sim.index(max(cos_sim))]
            self.replace_word.append(rpw)
            self.rpl_vectors.append(self.model.wv[rpw])
            cal = self.cos_sim(self.model.wv.get_vector(list(self.candidate_word_set)[cos_sim.index(max(cos_sim))]),self.model.wv.get_vector(targ_wrd))
            theta = 0.7
            if(cal>theta):
                self.metaphor.append("literal")
                self.gen_text.append(txt)
            else:
                self.metaphor.append("metaphor")
                self.gen_text.append(txt.replace(targ_wrd, rpw))
                
      else:
            self.metaphor.append("misc")
            self.gen_text.append(txt)
            self.replace_word.append("")
            self.rpl_vectors.append("")

    def summarizer(self):
      vec=[]
      cs=[]
      sent_tok=[]
      for st in self.gen_text:
        sent_tok.append(nltk.word_tokenize(st))
      for st in sent_tok:
        vec.append(self.get_sif_feature_vectors(st,sent_tok))
      for v in vec:
        c=[]
        for b in vec:
          c.append(self.cos_sim(v,b))
        cs.append(np.array(c))
      sentence_similarity_graph = nx.from_numpy_array(np.array(cs))
      scores = nx.pagerank(sentence_similarity_graph)
      ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(self.gen_text)), reverse=True)
      sum=[]
      for i in range(int(len(self.gen_text)/2)):
          sum.append("".join(ranked_sentence[i][1]))
      self.summary="".join(sum)

    def abstractive_summarizer(self):
      text = " ".join(self.gen_text)
      model = T5ForConditionalGeneration.from_pretrained('t5-small')
      tokenizer = T5Tokenizer.from_pretrained('t5-small')
      device = torch.device('cpu')
      preprocess_text = text.strip().replace("\n","")
      t5_prepared_Text = "summarize: "+preprocess_text
      tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
      # summmarize 
      summary_ids = model.generate(tokenized_text,
                                          num_beams=4,
                                          no_repeat_ngram_size=2,
                                          min_length=30,
                                          max_length=100,
                                          early_stopping=True)

      output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

      self.abstractive_summary = output
     

    def fit(self,sentences):
      print("Training the model............")
      self.abstract_score_preprocessing()
      self.pcaf()
      self.find_target_words(sentences)
      
      for sentence in self.sentences_tokenized:
          self.context_vectors.append(self.compute_context_vector(sentence))
      
      for i in range(len(self.context_vectors)):
        if len(self.context_vectors)>1:
                final_context_vector,idx = self.find_context_sims(self.context_vectors[i])
                self.f_con_vec.append(final_context_vector)
                context_vector = [final_context_vector]
        else:
          self.f_con_vec.append(self.context_vectors[i])
          context_vector = self.context_vectors[i]
        self.find_candidate_wordset(self.target_words[i])
        cos_sim = self.find_cos_sim(context_vector)
    
        self.predict_metaphor(cos_sim,self.target_words[i], nltk.sent_tokenize(sentences)[i])        

      self.summarizer()
      self.abstractive_summarizer()
      print("Training finished!")

