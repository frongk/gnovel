from bs4 import BeautifulSoup

from nltk import word_tokenize, sent_tokenize, pos_tag

from tqdm import tqdm
from collections import Counter
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

import pdb


# import mobi
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
mobidir = os.path.join(parentdir, 'gnovel/mobi')
sys.path.insert(0,mobidir) 
from mobi import Mobi


class Book(object):
    def __init__(self, book_path):
        self.all_words = self.collect_book_tokens(book_path)
    
    def collect_book_tokens(self, book_path):
        book = Mobi(book_path)
        book.parse()
        
        records = []
        sentences = []
           
        all_words = []
        for record in tqdm(book, desc='record_no'):
            record = record.decode('utf-8','replace')
            
            for item in BeautifulSoup(record, 'lxml').find_all('p'):
                block = item.text.lower()
        
                tokens = word_tokenize(block)
                all_words += tokens
    
        return all_words

class Characters(object):

    def __init__(self, character_file_path):
        self.characters, self.multiples = self.get_character_names(character_file_path)
    
    def get_character_names(self, character_path):
        # get character names ready
        with open(character_path, 'rb') as fi:
            characters = [ii.replace('\r\n','').lower().split(' ') for ii in fi.readlines()]
            # manually remove some minor characters
            multiple_ = Counter(list(itertools.chain.from_iterable(characters)))
            multiples = [key for key in multiple_ if multiple_[key]>1 ]
        
        return characters, multiples

class InteractionProbability(object):

    def __init__(self, book_obj, character_obj, window=30):
        self.count_df = self.count_appearances(book_obj.all_words, character_obj.characters, character_obj.multiples, window)
        self.p_intersection, self.p_conditional = self.interaction_matrix(self.count_df)

    def character_match(self, tokens, characters, multiples):
        token_set = set(tokens)
        token_set = token_set - set(multiples)
        
        match_vect = [len(token_set.intersection(set(cc))) for cc in characters]
        return np.array(match_vect, dtype=np.int)
    
    
    def count_appearances(self, all_words, characters, multiples, window=30):
    
        count_vects = np.empty([len(characters), len(all_words)], dtype=np.int)
        
        for start_idx in tqdm(range(len(all_words)-window), desc='counting occurrences'):
            focus = all_words[start_idx:start_idx+window]
            
            match_vector = self.character_match(focus, characters, multiples)
            count_vects[:,start_idx]=match_vector
        
        character_index = [' '.join(vect) for vect in characters]
        count_df = pd.DataFrame(count_vects, index=character_index)
        count_df = count_df.loc[count_df.index[count_df.sum(axis=1)!=0]]

        return count_df
    
    
    
    def get_interaction(self, count_df, p1, p2):
        p1_set = set(count_df.columns[count_df.loc[p1]>0])
        p2_set = set(count_df.columns[count_df.loc[p2]>0])
        return len(p1_set.intersection(p2_set))
    
    def interaction_matrix(self, count_df):
    
        p_intersection = pd.DataFrame(
                                      np.empty([count_df.shape[0], count_df.shape[0]]), 
                                      index=count_df.index, 
                                      columns = count_df.index
                                     )
        
        p_conditional = p_intersection.copy()
    
        sum_all = float(count_df.sum().sum())
        individual_prob = count_df.sum(axis=1)/sum_all
    
        for char1 in tqdm(count_df.index, desc="counting intersections"):
            for char2 in count_df.index:
                if char1 != char2:
        
                    c1n2 = self.get_interaction(count_df, char1, char2)
                    p1n2 = c1n2 / sum_all
        
                    if individual_prob.loc[char2] !=0:
                        p1c2 = p1n2 / individual_prob.loc[char2]
                    else:
                        p1c2 = 0
        
                    p_intersection[char1][char2] = p1n2
                    p_conditional[char1][char2] = p1c2
    
                else:
                    p_intersection[char1][char2] = 0
                    p_conditional[char1][char2] = 0
                
        return p_intersection, p_conditional
     

class BookGraph(object):

    def __init__(self, df, file_prefix, popularity_cutoff=None, count_vects=None, fig_size=(7,7), cut=0.95):

        if popularity_cutoff is None:
            self.df = df
        else:
            try:
                self.df = self.get_popular_df(df, count_vects, popularity_cutoff)
            except:
                print "error: make sure to pass both popularity_cutoff and count_vects dataframe!"

        
        self.file_prefix = file_prefix
        self.fig_size=fig_size

        # for repositioning graph to fit in plot window
        self.cut = cut

    def plot_all(self, df):

        self.pcolor_vis(df)
        self.draw_graph(df)

    def get_popular_df(self, df, count_vects, popularity_cutoff):
        pop_idx =  count_vects.sum(axis=1).sort_values(ascending=False)[:popularity_cutoff].index
        return df[pop_idx].loc[pop_idx]
        
    def pcolor_vis(self, matrix):

        plt.figure(figsize=self.fig_size, facecolor="w", frameon=False)
        plt.pcolor(matrix)
        plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index)
        plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns, rotation='vertical')
        plt.tight_layout()
        plt.savefig(self.file_prefix + '_pcolor.png')
        plt.close()
        
    def draw_graph(self, matrix):

        plt.figure(figsize=self.fig_size, facecolor="w", frameon=False)
        nvp_graph = [(index, column, matrix[column][index]) for index in matrix.index for column in matrix.columns]
        nvp_clean = [row for row in nvp_graph if row[2] > 0]
        nvp_df = pd.DataFrame(nvp_clean, columns=['vertex1','vertex2','edge_weight'])
        nvp_df = nvp_df[nvp_df['edge_weight'] > (nvp_df['edge_weight'].mean() - \
                        nvp_df['edge_weight'].std())]
        
        G = nx.from_pandas_edgelist(nvp_df, 'vertex1', 'vertex2', 'edge_weight')
        edges,weights = zip(*nx.get_edge_attributes(G,'edge_weight').items())
        
        # pos = nx.spring_layout(G)
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G)
        
        weights = [200*w for w in weights]
        
        plt.figure(figsize=self.fig_size, facecolor="w", frameon=False)
        nx.draw(G, pos=pos, node_color='#ae3333', node_size=500, edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.autumn)
        
        offset = 0.06
        pos_labels = {}
        keys = pos.keys()
        for key in keys:
            x, y = pos[key]
            pos_labels[key] = (x, y+offset)
        nx.draw_networkx_labels(G, pos=pos_labels, fontsize=12, font_family='Arial')

        cut = self.cut
        xmin= cut*min(xx for xx,yy in pos.values())
        ymin= cut*min(yy for xx,yy in pos.values())
        xmax= cut*max(xx for xx,yy in pos.values())
        ymax= cut*max(yy for xx,yy in pos.values())
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)

        plt.savefig(self.file_prefix + '_graph.png')
        plt.close()

class Gnovel(object):
    # autogenerate book graph 
    def __init__(self, book_path, character_path, intersectvconditional='intersection', popularity_cutoff=20):
        self.book = Book(book_path)
        self.characters = Characters(character_path)
        self.imat = InteractionProbability(self.book, self.characters)
        self.count_vects = self.imat.count_df

        if intersectvconditional == 'intersection':
            self.matrix = imat.p_intersection
        elif intersectvconditional == 'conditional' :
            self.matrix = imat.p_conditional

        self.popularity_cutoff = popularity_cutoff
    
    def make_graph(self, file_prefix, fig_size=(9,8),cut=1.35):
        bgraph = BookGraph(self.matrix, self.file_prefix, popularity_cutoff=self.popularity_cutoff, count_vects=self.count_vects, fig_size=fig_size, cut=cut)
        bgraph.draw_graph(bgraph.df)
        

if __name__ == "__main__":
    
    hp1_p = "books/Harry Potter and the Sorcerer's Stone - J. K. Rowling.mobi"
    hp2_p = "books/Harry Potter and the Chamber of Secrets - J. K. Rowling.mobi"
    hp3_p = "books/Harry Potter and the Prisoner of Azkaban - J. K. Rowling.mobi"
    hp4_p = "books/Harry Potter and the Goblet Of Fire - J. K. Rowling.mobi"
    hp5_p = "books/Harry Potter and the Order of the Phoenix - J. K. Rowling.mobi"
    hp6_p = "books/Harry Potter and the Half-Blood Prince - J. K. Rowling.mobi"
    hp7_p = "books/Harry Potter and the Deathly Hallows - J. K. Rowling.mobi"
    
    book_set =[hp1_p, hp2_p, hp3_p, hp4_p, hp5_p, hp6_p, hp7_p]
    for idx, book in enumerate(book_set):
        hp1 = Book(book)
        
        hp_char_path = 'hp_characters_edit.txt'
        people = Characters(hp_char_path)
           
        imat = InteractionProbability(hp1, people)
        p_int = imat.p_intersection
        p_cond = imat.p_conditional
        # df = imat.count_df
        
        bgraph = BookGraph(p_int, file_prefix='graphs/hp'+str(idx+1)+'_int', popularity_cutoff=20, count_vects=imat.count_df, fig_size=(9,8),cut=1.35)
        bgraph.draw_graph(bgraph.df)

        bgraph = BookGraph(p_int, file_prefix='graphs/big_hp'+str(idx+1)+'_int', popularity_cutoff=None, count_vects=imat.count_df, fig_size=(24,20),cut=1.05)
        bgraph.draw_graph(bgraph.df)
    
    
