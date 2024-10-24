import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")

class FinGuide:
    def __init__(self):
        self.urls = [
            'https://www.cbsl.gov.lk/en/financial-system/financial-markets/government-securities-market',
            'https://www.sc.com/ke/investments/learn/understanding-bonds-for-beginners/',
            'https://www.researchgate.net/publication/275543195_Treasury_Bills_and_Central_Bank_Bills_for_Monetary_Policy',
            'https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13376',
            'https://www.richmondfed.org/-/media/richmondfedorg/publications/research/special_reports/instruments_of_the_money_market/pdf/chapter_07.pdf',
            'https://drive.google.com/file/d/19QbWMrI9KjFjVHQu8Un00Y2gz49K-8Y3/view?usp=sharing'
        ]
        self.docs = None
        self.db = None
        self.llm_model = None
        self.rag_chain = None

    def load_data(self):
        loader = UnstructuredURLLoader(urls=self.urls)
        data = loader.load()
        print(f"Loaded {len(data)} documents")
        return data

    def process_text(self, data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.docs = text_splitter.split_documents(data)
        print(f"Split into {len(self.docs)} chunks")

    def create_embeddings_and_index(self):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        self.db = FAISS.from_documents(self.docs, embeddings)
        print("Created FAISS index")

    def setup_llm(self):
        self.llm_model = HuggingFaceHub(
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            model_kwargs={"temperature": 0.9, 'max_length': 180}
        )

    def setup_rag_chain(self):
        template = """You are assistant for a financial institution. Use the following information to answer the questions. If you don't know the answer, just say that you don't know. Use 10 sentences maximum to answer each question and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
        
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm_model
            | StrOutputParser()
        )

    def answer_question(self, question: str) -> str:
        return self.rag_chain.invoke(question).split("Answer:")[-1]

    def load_predicted_rates(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def generate_investment_advice(self, rates_df: pd.DataFrame) -> str:
        context = rates_df.to_string(index=False)
        question = "Based on the predicted future interest rates, what is the best time for investors to invest in treasury bills and bonds? Please provide some advice."
        
        formatted_prompt = f"""
        Context: {context}
        Question: {question}
        """
        
        return self.answer_question(formatted_prompt)

app = Flask(__name__)
CORS(app)

# Initialize FinGuide instance
fin_guide = FinGuide()

# Load data and process it once on startup
data = fin_guide.load_data()
fin_guide.process_text(data)
fin_guide.create_embeddings_and_index()
fin_guide.setup_llm()
fin_guide.setup_rag_chain()

@app.route('/')
def home():
    return jsonify(message="Welcome to FinGuide API"), 200

@app.route("/answer", methods=["POST"])
def answer_question():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = fin_guide.answer_question(question)
    return jsonify({"answer": answer})

@app.route("/investment_advice", methods=["POST"])
def generate_investment_advice():
    file_path = request.json.get("file_path")
    if not file_path:
        return jsonify({"error": "No file path provided"}), 400
    
    rates_df = fin_guide.load_predicted_rates(file_path)
    advice = fin_guide.generate_investment_advice(rates_df)
    
    return jsonify({"advice": advice})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)