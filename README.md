# gnovel
For Automatic Generation of Character Maps Using NLP

## Example Output

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
