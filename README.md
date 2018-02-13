# gnovel
For Automatic Generation of Character Maps Using NLP

## Example Output
![Harry Potter 1](https://raw.githubusercontent.com/frongk/gnovel/master/harrypotter_orig/graphs/hp1_int_graph.png)
## Usage and Setup

### Usage
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
