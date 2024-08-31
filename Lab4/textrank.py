

from summa import keywords
from summa.summarizer import summarize

text1 = "Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organizedrepresentation in many countries and international networks. It borrows elementsophobia, anti-Romanyism, antisemitism, anti-communism and initiating the FourthReich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries(especially Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to theideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia,anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration ofAdolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism."

text2 = "Prototypes are widely recognized to be a core means of exploring and expressing designs for interactive computer artifacts. It is common practice to build prototypes in order to represent different states of an evolving design, and to explore options. How- ever, since interactive systems are complex, it may be difficult or impossible to create prototypes of a whole design in the formative stages of a project. Choosing the right kind of more focused prototype to build is an art in itself, and communicating its limited purposes to its various audiences is a criti- cal aspect of its use.The ways that we talk, and even think about pro- totypes, can get in the way of their effective use. Current terminology for describing prototypes cen- ters on attributes of prototypes themselves, such as what tool was used to create them, and how re- fined-looking or -behaving they are. Such terms can be distracting. Tools can be used in many dif- ferent ways, and detail is not a sure indicator of completeness.We propose a change in the language used to talk about prototypes, to focus more attention on fun- damental questions about the interactive system being designed: What role will the artifact play in a user’s life? How should it look and feel? How should it be implemented? The goal of this chapter is to establish a model that describes any prototype in terms of the artifact being designed, rather than the prototype’s incidental attributes. By focusing on the purpose of the prototype—that is, on what it prototypes—we can make better decisions aboutThis article is published, in a different format, as Houde, S., and Hill, C., What Do Prototypes Prototype?, in Handbook of Human-Computer Interaction (2nd Ed.), M. Helander,T.␣ Landauer, and P. Prabhu (eds.): Elsevier Science B. V: Amsterdam, 1997.the kinds of prototypes to build. With a clear pur- pose for each prototype, we can better use proto- types to think and communicate about design.In the first section we describe some current diffi- culties in communicating about prototypes: the complexity of interactive systems; issues of multi- disciplinary teamwork; and the audiences of pro- totypes. Next, we introduce the model and illus- trate it with some initial examples of prototypes from real projects. In the following section we present several more examples to illustrate some further issues. We conclude the chapter with a sum- mary of the main implications of the model for prototyping practice."

text3 = "There are around fifty survey articles published in the recent years that deal with 4G and 5G cellular networks. From these survey articles only seven of them deal with security and privacy issues for 4G and 5G cellular networks and none of the previous works covers the authentication and privacy preserving issues of 4G and 5G networks. The article “” is the first on the literature that thoroughly covers authentication and privacy preservation threat models, countermeasures, and schemes that we recently proposed from the research community. It suggests that …Security is an important part of the 4G system as well and many aspects are in fact quite similar in 4G and 5G systems. The article “” highlights several Firstly, both 4G and 5G systems use mutual authentication between the User Equipment (UE) and the network. The network thus authenticates both the UE and the network, which is a two-way authentication process that prevent unauthorized access. Secondly, both 4G and 5G networks supports ciphering, which is ….However, there are also some differences in security between the networks. Firstly, unlike 4G networks, where exceptions allowed the permanent subscriber identifier (IMSI) to be transmitted in clear text, 5G networks ensure that the permanent subscription identifier is never sent in clear text over the air. Secondly, as mentioned earlier, 5G networks supports ciphering. However, it also supports integrity protection and replay protection of Non-Access Stratum (NAS) signaling between the UE and the network. These mechanisms ensure that data exchanged between the user’s device and the network is encrypted, protected from tampering, and guarded against replay. In 4G networks, only ciphering was supported, and integrity protection was not as comprehensive. Thirdly, 5G networks provide greater configurability of security features. This means that the network can dynamically decide which security features, such as ciphering and integrity protection, should be used for a specific session. In 4G networks, User Plane security was typically activated in all cases, even when ciphering was not required. "

# First text
# Define length of the summary as a proportion of the text
print(summarize(text1, ratio=0.2))
print("\n")
summarize(text1, words=30)

# Keyword extraction
print("Keywords:\n", keywords.keywords(text1))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n", keywords.keywords(text1, words=3))
print("\n")
# Second text
# Define length of the summary as a proportion of the text
print(summarize(text2, ratio=0.1))
print("\n")
summarize(text2, words=30)

# Keyword extraction
print("Keywords:\n", keywords.keywords(text2))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n", keywords.keywords(text2, words=2))
print("\n")

# Third text
# Define length of the summary as a proportion of the text
print(summarize(text3, ratio=0.2))
print("\n")
summarize(text3, words=30)

# Keyword extraction
print("Keywords:\n", keywords.keywords(text3))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n", keywords.keywords(text3, words=2))
print("\n")
