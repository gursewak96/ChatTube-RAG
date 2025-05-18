# ChatTube 🗣️📺: Your Interactive YouTube Video Companion

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-orange.svg)](https://www.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20GPT--4o--mini-brightgreen.svg)](https://openai.com/)

Ever felt overwhelmed by lengthy YouTube videos like podcasts or lectures? ChatTube is here to help! It's a RAG (Retrieval Augmented Generation) system that lets you "chat" with YouTube videos, ask questions, and get summaries without watching the entire thing. Save time and get to the core of the content, faster! 🚀

## The Problem 😕

Many valuable YouTube videos are long. Finding specific information or just getting a gist can mean:
*   Spending hours watching the whole video.
*   Endlessly scrubbing through the timeline.
*   Missing out on key insights due to time constraints.

This makes learning and information retrieval inefficient.

## The Solution ✨: ChatTube!

ChatTube transforms your YouTube experience by:
1.  Fetching the video's transcript.
2.  Processing and understanding the content using cutting-edge AI.
3.  Allowing you to:
    *   ❓ **Ask specific questions** and get relevant, context-aware answers.
    *   📝 **Request summaries** of the video or specific parts.
    *   💡 **Explore topics** discussed without needing to watch from start to finish.

It's built using Python, LangChain, and OpenAI's powerful language models.

## Key Features 🚀

*   **Interactive Q&A:** Get answers directly from the video's content.
*   **Smart Summarization:** Understand long videos in minutes.
*   **Efficient Information Retrieval:** No more manual searching!
*   **Powered by LangChain & OpenAI:** Leverages state-of-the-art RAG techniques.
*   **Handles Missing Transcripts:** Gracefully informs if a video has no captions.

## How It Works (The RAG Magic) 🪄

The system follows a Retrieval Augmented Generation (RAG) pipeline:

1.  📜 **Transcript Acquisition:**
    *   Given a YouTube video ID, it fetches the English transcript using `youtube-transcript-api`.
    *   If transcripts are disabled, it informs the user.

2.  🧩 **Text Splitting (Chunking):**
    *   The full transcript is broken down into smaller, manageable chunks (e.g., 1000 characters with overlap) using `RecursiveCharacterTextSplitter`. This helps in targeted retrieval.

3.  🧠 **Embedding Generation & Storage:**
    *   Each chunk is converted into a numerical representation (embedding) using OpenAI's `text-embedding-3-small` model.
    *   These embeddings are stored in a `FAISS` vector store, which allows for efficient similarity searches.

4.  🔍 **Retrieval:**
    *   When you ask a question (your "query"), it's also embedded.
    *   The `FAISS` vector store retrieves the most relevant chunks from the transcript based on semantic similarity to your query (e.g., top 4 similar chunks).

5.  🗣️ **Augmentation & Prompting:**
    *   The retrieved chunks (the "context") and your original question are combined into a carefully crafted prompt.
    *   This prompt guides the Language Model (LLM) to answer based *only* on the provided video context.

6.  💬 **Generation (The Answer!):**
    *   The prompt is sent to an OpenAI LLM (`gpt-4o-mini` with low temperature for factual answers).
    *   The LLM generates a response based on the context and question.

7.  🔗 **Chaining with LangChain Expression Language (LCEL):**
    *   The entire process is elegantly chained together using LCEL components:
        *   `RunnableParallel`: To fetch context and pass the question simultaneously.
        *   `RunnableLambda`: To format the retrieved documents into a string.
        *   `RunnablePassthrough`: To carry the original question along the chain.
        *   `PromptTemplate`: To structure the input for the LLM.
        *   `ChatOpenAI`: The LLM itself.
        *   `StrOutputParser`: To get a clean string output from the LLM.

## Tech Stack 🛠️

*   **Core Language:** Python
*   **YouTube Interaction:** `youtube-transcript-api`
*   **Orchestration:** `LangChain` (specifically `langchain-community`, `langchain-openai`, `langchain-core`)
    *   Text Splitting: `RecursiveCharacterTextSplitter`
    *   Embeddings: `OpenAIEmbeddings` (model: `text-embedding-3-small`)
    *   Vector Store: `FAISS` (CPU version: `faiss-cpu`)
    *   LLM: `ChatOpenAI` (model: `gpt-4o-mini`)
    *   Prompting: `PromptTemplate`
    *   LCEL: `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `StrOutputParser`
*   **Tokenization (for OpenAI):** `tiktoken`
*   **Environment Management:** `python-dotenv` (for API keys)

## Dive into the Code (Jupyter Notebook Highlights) 📓

The provided Jupyter Notebook walks through these steps:

1.  **Setup:**
    *   Installs necessary libraries:
        ```python
        !pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv
        ```
    *   Imports required modules.

2.  **Step 1: Indexing (Document Ingestion & Processing)**
    *   `1a - Get Transcript`: Fetches video transcript using `video_id`.
        ```python
        video_id = "Gfr50f6ZBvo" # Example video
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        ```
    *   `1b - Text Splitting`: Chunks the transcript.
        ```python
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        ```
    *   `1c & 1d - Embedding & Storing`: Generates embeddings and stores them in FAISS.
        ```python
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        ```

3.  **Step 2: Retrieval**
    *   Creates a retriever from the vector store.
        ```python
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        ```

4.  **Step 3: Augmentation**
    *   Initializes the LLM and defines the prompt template.
        ```python
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = PromptTemplate(...) # As defined in the notebook
        ```
    *   (Manual way shown first) Retrieves docs, formats context, and invokes the prompt.

5.  **Step 4: Generation**
    *   (Manual way shown first) Gets the answer from the LLM.
        ```python
        answer = llm.invoke(final_prompt)
        print(answer.content)
        ```

6.  **Building the Chain with LCEL ⛓️**
    *   Introduces `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, and `StrOutputParser`.
    *   Defines a helper function `format_docs`.
    *   Constructs the full RAG chain:
        ```python
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        # Example Invocation
        response = main_chain.invoke('Can you summarize the video')
        print(response)
        ```

## To Run This Notebook 🧑‍💻

1.  **Prerequisites:**
    *   Python 3.8+ and Pip.
    *   An OpenAI API Key.

2.  **Installation:**
    Open a terminal or command prompt and run:
    ```bash
    pip install youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv jupyterlab
    ```

3.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the same directory as your notebook.
    *   Add your OpenAI API key to it:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   The notebook will use `python-dotenv` to load this key.

4.  **Run the Jupyter Notebook:**
    *   Launch Jupyter Lab or Jupyter Notebook:
        ```bash
        jupyter lab
        # or
        jupyter notebook
        ```
    *   Open the `.ipynb` file.
    *   You can change the `video_id` in the notebook to experiment with different videos.
    *   Run the cells sequentially.

## Example Interaction 💬

**User:** "What did they say about the future of AI in this video?"
*(Assuming the video `Gfr50f6ZBvo` discusses AI futures)*

**ChatTube:** "In the video, the speaker mentioned that the future of AI is likely to involve more advanced reasoning capabilities and wider integration into daily tasks. They also touched upon the importance of ethical guidelines as AI becomes more powerful." *(This is a hypothetical answer based on the RAG process)*

## Why This Project? 🤔

*   **Practical Application of RAG:** Demonstrates a real-world use case for Retrieval Augmented Generation.
*   **Efficiency Boost:** Solves a common problem of information overload with long-form video content.
*   **LangChain Showcase:** Highlights the power and flexibility of the LangChain framework for building LLM applications.
*   **Learning & Exploration:** A great way to understand how different components (embeddings, vector stores, LLMs, prompt engineering) work together.

## Future Ideas 🔮

*   ✨ **Streamlit/Gradio UI:** Create a user-friendly web interface.
*   🕒 **Timestamped Answers:** Link answers back to specific moments in the video.
*   🗣️ **Support for Other Languages:** Extend transcript capabilities.
*   💾 **Caching:** Cache processed videos to speed up repeated queries.
*   ⚙️ **More Advanced Retrieval:** Experiment with different retriever types or re-ranking.

## Connect with Me! 🔗

I'm passionate about building useful AI applications. Let's connect!
*   **LinkedIn:** [Your LinkedIn Profile URL]
*   **GitHub:** [Your GitHub Profile URL]