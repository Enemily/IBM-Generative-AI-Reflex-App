import reflex as rx
import os
from dotenv import load_dotenv

from genai.model import Credentials
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from typing import Any, List, Mapping, Optional, Dict
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Extra
try:
    from langchain import PromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
    from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace embedding models
    from langchain.text_splitter import CharacterTextSplitter #text splitter
    from langchain.llms.base import LLM
    from langchain.llms.utils import enforce_stop_tokens
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")

############  GET CREDENTIALS
load_dotenv()
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", 'https://us-south.ml.cloud.ibm.com')
project_id = os.getenv("PROJECT_ID", None)
if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }
############  


############ INSTANTIATE MODEL + LANGCHAIN INTERFACE
model_params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.9,
    GenParams.TOP_K: 100,
    GenParams.TOP_P: 0.3,
    GenParams.REPETITION_PENALTY: 2.0    
}

# Instantiate a model proxy object to send your requests
llm = Model(
    model_id='google/flan-ul2',
    params=model_params,
    credentials=creds,
    project_id='')

class LangChainInterface(LLM, BaseModel):
    credentials: Optional[Dict] = None
    model: Optional[str] = None
    params: Optional[Dict] = None
    project_id : Optional[str]=None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _params = self.params or {}
        return {
            **{"model": self.model},
            **{"params": _params},
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM WATSONX"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the WatsonX model"""
        params = self.params or {}
        model = Model(model_id=self.model, params=params, credentials=self.credentials, project_id=self.project_id)
        text = model.generate_text(prompt)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

model = LangChainInterface(model='google/flan-ul2', params=model_params, credentials=creds, project_id = project_id)
############ 


############ STATE CLASS
class State(rx.State):
    """The app state."""

    img: list[str]
    question: str = ""
    chat_history: list[tuple[str,str]]


    ############ FUNCTION TO GENERATE ANSWER TO USER'S QUESTION ABOUT THE CURRENT PDF (so lanchain inference is on previous file data)
    async def answer(self): 
        # create new QnA pair
        answer = ""
        self.chat_history.append((self.question, answer))
        
        # write file to server
        outfile = f".web/public/{self.img[-1]}"

        # langchain 
        loaders = [PyPDFLoader(outfile)]
        loaders.append(PyPDFLoader(outfile))
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)
        
        chain = RetrievalQA.from_chain_type(llm=model, 
                                    chain_type="stuff", 
                                    retriever=index.vectorstore.as_retriever(), 
                                    input_key="question")
        
        # get lanchain response according to user's question
        chainResponse = chain.run(self.question)
        self.chat_history[-1] = (self.question, chainResponse)
        yield
    ############
    

    ############ FUNCTION TO GENERATE ANSWER TO USER'S QUESTION ABOUT A NEW PDF (so create new langchain inference according to new file laoded)
    async def handle_upload(
        self, files: list[rx.UploadFile]
    ):
        # create new QnA pair
        answer = ""
        self.chat_history.append((self.question, answer))
        
        loaders = []
        
        for file in files:
            upload_data = await file.read()
            outfile = f".web/public/{file.filename}"

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)

            loaders = [PyPDFLoader(outfile)]
            loaders.append(PyPDFLoader(outfile))
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(),
                text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)
            
            chain = RetrievalQA.from_chain_type(llm=model, 
                                        chain_type="stuff", 
                                        retriever=index.vectorstore.as_retriever(), 
                                        input_key="question")
            
            
            chainResponse = chain.run(self.question)
            self.chat_history[-1] = (self.question, chainResponse)
            yield
    ############

############