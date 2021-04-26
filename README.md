# NLP_Project
## Figurative text inference leveraging metaphor interpretation and summarization

Information  retrieval,  though  a  classical  researchproblem  in  NLP,  continues  to  be  highly  relevant  due  to  thevariety  of  domain  based  challenges  associated  with  it.  One  ofthe   many   forms   of   information   retrieval   from   text   includesextractive summarization. Widely used in news applications suchas  inShorts  to  encapsulate  news  content.  In  spite  of  advancesmade  in  this  field,  challenges  in  key  areas  such  as  semanticinformativeness, continuity of language remain. Recent advancesin  the  field  of  figurative  text,  particularly  in  identification  andinterpretation  of  text  involving  metaphorical  usage  of  languageprovides  an  interesting  standpoint  to  approach  analysis  andinference  from  text.  In  this  project  we  propose  to  study  theinterplay  of  figurative  text  in  day  to  day  language  and  developa  framework  to  gauge  the  unsupervised  methods  of  metaphoridentification and interpretation to improve the text summariza-tion  quality  with  focus  on  semantic  informativeness.  We  focuson  text  formats  involving  small  sized  passage  text  like  newsarticles, poems, excerpts from books and aim to extract intendedmeaning based on the text. We position the success of our modelas  a  domain  independent  approach  which  can  find  utility  in  avariety  of  contexts  such  as  opinion  mining,  threat  identificationin  surveillance,  a  means  to  measure  intensity  of  emotion  etc.


## Installation process

1) Download the entire project from the git repository.
2) Open the project folder in an IDE or editor like VSCode or you could use the command line as well(path being inside the project folder). Make sure you have python installed and updated pip
3) In the command line of the editor or ide type "python -m venv <path to the environment>" to create an environment where "<path to the environment>" should be a path to your environment. After that activate the env by typing in the .\<path to the environment>\Scripts\activate
4) Now, unzip the wiki_word2vec.zip folder inside the model folder and copy all the files in the unzipped folder to the model folder.
5) Type python on the command line and type the following:

##to import the package
from model import inference_gen

##to create an object
m = inference_gen()

##to run the unsupervised model for the given text
m.fit("Paragraph to be tested")

##to find the target words of each sentence
m.target_words

##to print the words considered as replace words for the target words
m.replace_word

##to see the sentence tokenized version of the input para along with the replaced words for identified metaphors in every sentence
m.gen_text

##to print the identified metaphors for the given number of sentences in the above listed code's ouput
m.metaphor

##to print the extractive summary
m.summary

##to print the abstractive summary
m.abstractive_summary