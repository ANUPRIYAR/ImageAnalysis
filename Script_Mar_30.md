
"Welcome everyone! Today, I'm excited to talk about Large Language Models and how we can maximize their efficiency through advanced techniques.

In this presentation, I'll walk you through several advanced techniques that can dramatically improve LLM performance and efficiency. 
We'll explore everything from the foundations of how these models work to cutting-edge optimization approaches that can increase its efficency to a even greater extent.

#### Slide 2: Journey Overview
"Let's take a moment to map out our journey today. We'll be covering five key areas that build upon each other to give you a comprehensive understanding of LLM optimization.
First, we'll start with the Foundations of Tokenization - understanding how LLMs break down and process language at its most basic level. 
We'll explore both word-level and subword-level tokenization approaches and why the transition to subword methods has been so critical.

Second, we'll dive into Advanced Tokenization Techniques like Byte Pair Encoding, which powers models like GPT and Llama, and 
the WordPiece Algorithm which was developed by Google to be used by BERT to handle rare words more effectively.

Third, we'll examine Advanced Prompting Techniques, including traditional strategies like Chain-of-Thought and the revolutionary
Graph of Thoughts framework that visualizes how LLMs reason.  ---> Need to rephrase

Fourth, we'll look at Parameter-Efficient Techniques such as LoRA and QLoRA that optimize large models while minimizing computational costs. 
We'll also explore prompt and prefix tuning for specific applications in fields like legal and medical domains.
Finally, we will explore a Distillation technique called Knowledge distillation- 
(the process of compressing knowledge from massive models into smaller, more efficient ones, 
comparing traditional fine-tuning with modern approaches to help you determine which method to use for maximum impact.)

So I am hoping , By the end of this presentation, you'll have a solid understanding of how to unlock the full potential of LLMs for your specififc usecase 
Now, let's begin with the foundations of tokenization..."


### Slide 3: Impact of Tokenization in LLMs
"Now that we've mapped out our journey, let's dive into the fascinating impact of tokenization on LLMs. 
As you can see on this slide, tokenization isn't just a technical detail - it's at the root of many curious limitations you might have encountered when working with these models.
For example : Why can't LLMs spell words correctly sometimes? Tokenization. Why do they struggle with simple string processing tasks like reversing text?
Again, tokenization. The limitations with non-English languages, particularly character-based ones like Japanese? You guessed it - tokenization.
The pattern continues across various tasks. 
Simple arithmetic, Python coding challenges, and unexpected behaviors like abruptly halting when encountering specific strings like '<endoftext>' all trace back to how these models process text at the token level.
Even seemingly bizarre issues - like warnings about trailing whitespace, models breaking when encountering unusual compound words like 'SolidGoldMagikarp,'
or the preference for YAML over JSON - all stem from tokenization decisions.
And yes, all these issues can be attributed to the fact that how  Tokenization was performed in these models
This humorous but insightful slide highlights just how fundamental this process is to understanding both the capabilities and limitations of language models."


### Slide 4: Understanding Tokenization
"Let's break down the tokenization process to understand what's happening behind the scenes.
This flowchart illustrates the journey from raw input text to the final tokens that an LLM processes.
The process begins with input text that undergoes preprocessing - converting to lowercase, removing special characters, handling punctuation, and normalizing the text.
These steps prepare the text for consistent tokenization.


After preprocessing, <strong> we specify token delimiters which lead to different splitting approaches </strong> :
(For example we have GPT2 tokrnizer which prefixes a special charcter G in fron of each word, then it performs the tokenisation. The we have Sentence peice tokenizer which prepends a _ in front 
of each word.)
space splitting for traditional word tokenization, subword splitting 
which we'll explore in depth shortly, and character-level splitting for the most granular approach.
The tokenization algorithm itself can employ various methods: rule-based approaches following linguistic principles, statistical methods that learn from data patterns,
or dedicated machine learning models trained specifically for tokenization tasks.
Finally, we have post-processing steps like stopword removal or stemming and lemmatization, which further refine the tokens before they're ready for use in downstream NLP tasks.

So This entire pipeline determines how an LLM 'sees' text, which directly impacts its understanding and generation capabilities. The decisions made at each step create both the
strengths and weaknesses we observe in these models."


### Slide 5: Exploring Key Tokenization Techniques
"Now let's explore the key tokenization techniques and discuss when to use each approach.
So staring with the word level tokenisation is splitting based on space, will give you word level tokenisation.
Word-level tokenization has certain advantages - it's effective for simple text in languages with stable word forms like English, 
<strong> where words don't change much based on their usage </strong> . This approach enables faster preprocessing and reduced memory usage in some contexts.
However, the limitations are significant. Word-level tokenization struggles with handling rare or out-of-vocabulary words effectively- 
imagine encountering technical terms or names not seen during training. And also for the languages which are morphologically rich , it cannot handle them very effectively, 
 < strong> where words change form based on tense, plurality( or number), or other factors. < /strong> . And if we try to handle those , , it's can be very memory inefficient 
 because it will have very large vocablury as each unique word needs its own token.

So In order to handle the limtations of word level tokenisation we can use Sub word level tokenisation
(So sub word level tokenisation comes to the rescue here.)
This is why the transition to subword-level tokenization has been so revolutionary. In contexts where word-level approaches fall short, 
So in subword tokenization we can break the words into meaningful segments. 
As illustrated in the right panel, we can see how different approaches segment the same input text.
(This method is particularly beneficial for addressing the limitations we just discussed, especially with rare vocabulary words and complex linguistic structures.)
So there are many tokenisation algorithms which uses sub word tokenisation.
The most common techniques include Byte Pair Encoding (BPE), which powers models like GPT and Llama, and WordPiece, used by BERT - both of which we'll explore in detail in our next slides.
The visualization on the right shows how character tokenization, word tokenization, and subword tokenization each handle the same input differently,
demonstrating the trade-offs in granularity and meaning preservation."


###  Slide 6: Byte Pair Encoding (BPE)
"Now let's explore Byte Pair Encoding in depth - the tokenization algorithm powering many of today's most advanced language models including GPT and Llama.
What makes BPE so special? Let me walk you through it interactively. Imagine we're building a vocabulary from scratch, just as shown in the upper right corner of this slide.
We start with individual characters - our initial tokens.
Looking at our example with the word 'llama' repeated, BPE works by identifying the most frequent consecutive character pairs. 
In iteration 1, we see that 'am' appears most frequently, so we merge these characters into a single token. 
In iteration 2, 'ama' becomes common. By iteration 5, whole words like 'llama' have become single tokens!
This iterative merging process is precisely what gives modern LLMs their flexibility. 
The algorithm continuously counts the frequency of neighboring pairs, identifies the most frequent combinations, and merges them into new subword tokens
until reaching an optimal vocabulary size.
< strong > Let me pause here and ask: Have you noticed how this solves the out-of-vocabulary problem we discussed earlier? </strong > Even when encountering entirely new words,
BPE can break them down into familiar subwords, giving the model a fighting chance at understanding novel terms.
( so it will give the model a very good chnace to understand any new terms that comes up)
The diagram at the bottom illustrates this process at scale. When applied to large text corpora, BPE creates vocabularies that efficiently represent language
with a balance of common words as single tokens and rare words as combinations of subword pieces.
This is the secret sauce/ or technique  behind how models like GPT can understand and generate such a wide range of vocabulary despite having a fixed token set."

### Slide 7: WordPiece Tokenization Algorithm
"Moving from BPE, let's explore WordPiece - another subword tokenization approach that's particularly notable as the algorithm behind BERT, DistilBERT, and other transformer models you've likely used.
	• WordPiece is the tokenization algorithm Google developed to pretrain BERT
![image](https://github.com/user-attachments/assets/ba6707ec-b92d-4e71-8ed4-2d1d1f6759d5)

[[It’s very similar to BPE in terms of the training, but the actual tokenization is done differently.
	• Thus, the initial alphabet contains all the characters present at the beginning of a word and the characters present inside a word preceded by the WordPiece prefix.
	• Then, again like BPE, WordPiece learns merge rules. 
		○ The main difference is the way the pair to be merged is selected. Instead of selecting the most frequent pair, WordPiece computes a score for each pair, using the following formula:
![image](https://github.com/user-attachments/assets/4fc6eabd-d636-47f0-8a4e-8ed5dc98b333)
![image](https://github.com/user-attachments/assets/08e4633e-a99e-402e-9752-aeb5751a56f6)
	• By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary. 
![image](https://github.com/user-attachments/assets/586a4b88-161f-4294-86a3-6be61d11160a)


	•  Since it identifies subwords by adding a prefix (like ## for BERT), each word is initially split by adding that prefix to all the characters inside the word.
![image](https://github.com/user-attachments/assets/e77dd3de-1ca4-4e76-81e3-157f8566eaa0)
	• For instance, it won’t necessarily merge ("un", "##able") even if that pair occurs very frequently in the vocabulary, because the two pairs "un" and "##able" will likely each appear in a lot of other words and have a high frequency.
![image](https://github.com/user-attachments/assets/df5a706e-be5c-430c-b4ae-d8c502458a0f)  ]]

Similar to BPE, WordPiece breaks words into smaller meaningful units, but with some key differences in implementation that make it especially effective for handling rare words.


Let me make this concrete with an example. Look at the right side of the slide, where we tokenize the phrase 'WordPiece tokenization is a powerful technique in NLP.' Notice how 'tokenization' gets broken into 'token' and 'ization' - preserving meaningful linguistic units rather than arbitrary character combinations.
Here's how it works in four simple steps:

First, it starts with individual characters as tokens - just like BPE.
Then it iteratively merges the most frequent token pairs into new subwords.
It stops once a predefined vocabulary size is reached.
And critically, uncommon words get broken into smaller subword units.


Now, I'd like you to pay special attention to the advantages this brings:
[gesturing to each point with emphasis]

It handles rare and unseen words by decomposing them into known subwords
It achieves an efficient vocabulary size by finding the sweet spot between character-based and full-word tokenization
And importantly, it improves multilingual model performance by reusing subwords across languages

At the bottom, you can see BERT's specific tokenization process. Words like 'Playing' become ['Play', '##ing'] and 'NewYork' becomes ['New', '##York']. 
Those '##' prefixes are BERT's way of indicating that a subword piece appears in the middle or end of a word, not at the beginning.
What real-world impact does this have?
It means models using WordPiece can better understand technical jargon, names, and specialized vocabulary they've never seen in training - a critical 
capability for practical applications in fields like medicine, law, and science.
