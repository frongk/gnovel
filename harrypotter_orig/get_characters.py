import urllib2 as urllib
from bs4 import BeautifulSoup

import csv

url = 'https://en.wikipedia.org/wiki/List_of_Harry_Potter_characters'
response = urllib.urlopen(url)
soup = BeautifulSoup(response.read(), "html")
main = soup.find("div", {"id": "bodyContent"})
li = [ii.text for ii in main.find_all("li")]

# wikipedia is lame to parse 
cutoffidx = [ii for ii, item in enumerate(li) if item == "Book: Harry Potter"][0]
char_text = li[:cutoffidx]
char_init = [ii.split(u'\u2013')[0] for ii in char_text if ii.find(u'\u2013')!=-1]
char_refine = [ii.split('-')[0] for ii in char_text if ii.find('-') != -1 and ii.find(u'\u2013')==-1]

characters = []
for char_init in char_init + char_refine:
    character = ''

    for letter in char_init:
        if letter not in ['(', ')', u'\xa0', ',', '\"']:
            character += letter
    
    character = character.replace('a.k.a. ','')
    character = character.replace(u'n\xe9e ','')
    character = character.replace('/',' ')


    if character[-1] == ' ':
        characters.append(character[:-1])
    else:
        characters.append(character)

with open('hp_characters.txt', 'wb') as fo:
    fo.write('\r\n'.join(characters))
    
name_match = [name.lower().split(' ') for name in characters]


