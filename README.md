# gnovel
For Automatic Generation of Character Maps Using NLP

## Description and Example Output
This package (in progress) can be used to analyze a `.mobi` ebook and determine the relationships between characters. 
![Harry Potter 1](https://raw.githubusercontent.com/frongk/gnovel/master/harrypotter_orig/graphs/hp1_int_graph.png)

## Method
There are two inputs that are used the script. The ebook in `.mobi` format and a text list of the characters that you want to analyze where each character is separated by a newline character. These scripts will read through the ebook file and find instances in which different characters interact. This is tabulated as probabilities of characters co-appearing. After probabilities are generated, a graph depicting significant relationships is rendered.

## Usage and Setup
```
from gnovel.main import Gnovel

novel_graph = Gnovel('book.mobi', 'character_list.txt')
novel_graph.draw_graph('output.file')

```


### Dependencies
Make sure that your python environment has the following dependencies installed:
```
BeautifulSoup
nltk
tqdm 
numpy 
pandas
matplotlib
networkx

```
## Additional Features (Planned)
1. Character recognition using named-entity recognition (NER)
2. Relationship positivity (Sentiment Analysis)
