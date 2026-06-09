# Sample Documents for Self-Healing RAG

## Document 1: Climate Change
Climate change refers to long-term shifts in global temperatures and weather patterns. 
While some climate change is natural, since the 1800s human activities have been the main driver, 
primarily due to burning fossil fuels like coal, oil and gas. This produces greenhouse gases 
like CO2 and methane that trap heat in the atmosphere. Effects include rising sea levels, 
more extreme weather events, melting glaciers, and shifts in ecosystems.

## Document 2: Machine Learning
Machine learning is a subset of artificial intelligence that enables computers to learn from data 
without being explicitly programmed. Key types include supervised learning (learning from labeled data), 
unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through 
rewards and penalties). Popular algorithms include neural networks, decision trees, and support vector machines.

## Document 3: The Solar System
The solar system consists of the Sun and everything gravitationally bound to it. This includes 
eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. 
Jupiter is the largest planet, while Mercury is the smallest. Earth is the only known planet 
to harbor life. The solar system also contains dwarf planets like Pluto, asteroids, comets, and moons.

## Document 4: Python Programming
Python is a high-level, interpreted programming language known for its simplicity and readability. 
Created by Guido van Rossum in 1991, it supports multiple programming paradigms including 
procedural, object-oriented, and functional programming. Python is widely used in web development, 
data science, artificial intelligence, automation, and scientific computing. 
Popular libraries include NumPy, Pandas, TensorFlow, and Django.

## Document 5: The Human Brain
The human brain is the most complex organ in the body, containing approximately 86 billion neurons. 
It is divided into regions with specialized functions: the frontal lobe handles decision-making and 
personality, the temporal lobe processes language and memory, the parietal lobe manages sensory 
information, and the occipital lobe handles vision. The brain communicates via electrical and 
chemical signals called neurotransmitters.

## Document 6: Blockchain Technology
Blockchain is a distributed ledger technology that records transactions across multiple computers 
so records cannot be altered retroactively. Each block contains transaction data, a timestamp, 
and a cryptographic hash of the previous block, forming a chain. It was first described in 2008 
by Satoshi Nakamoto as the technology behind Bitcoin. Beyond cryptocurrency, blockchain is used 
in supply chain management, healthcare records, and voting systems.

## Document 7: World War II
World War II lasted from 1939 to 1945 and involved most of the world's nations. It began when 
Nazi Germany under Adolf Hitler invaded Poland on September 1, 1939. The war was fought between 
the Allies (including the US, UK, Soviet Union, and France) and the Axis powers (Germany, Italy, Japan). 
It ended in Europe on May 8, 1945 (V-E Day) and in the Pacific on September 2, 1945 (V-J Day) 
after the US dropped atomic bombs on Hiroshima and Nagasaki.

## Document 9: About This Assistant — Who Are You?
I am Self-Healing RAG, an intelligent AI assistant powered by a self-correcting Retrieval-Augmented Generation (RAG) pipeline.
Here is how I work: when you ask me a question, I first retrieve the most relevant information from my knowledge base using a FAISS vector index.
I then generate an answer using the qwen2.5-coder:7b language model running locally via Ollama.
A built-in critic node then evaluates whether my answer is actually grounded in the retrieved context.
If the answer fails the critique, I automatically rewrite the search query and try again — up to 2 times — before giving you a final response.
This self-healing loop means I correct my own mistakes before you ever see them.
I can answer general knowledge questions, assist with coding tasks, generate and run Python code, and learn new information you teach me.
My knowledge base includes topics such as climate change, machine learning, the solar system, Python programming, the human brain, blockchain, World War II, and DNA/genetics.
I was built with FastAPI, LangGraph, LangChain, HuggingFace embeddings, and FAISS.

## Document 8: DNA and Genetics
DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for development, 
functioning, growth and reproduction of all known organisms. It consists of two strands forming 
a double helix, made up of four bases: adenine (A), thymine (T), guanine (G), and cytosine (C). 
Genes are segments of DNA that encode proteins. Humans have approximately 20,000-25,000 genes 
contained in 23 pairs of chromosomes.
