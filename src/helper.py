
def  file_process(file_path):
    """This function will take file path
    and return a list of documents splitted from the PDF file using LangChain.

    Args:
        file_path (_type_): file path where file is located .
    """
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loder=PyPDFLoader(file_path)
    data = loder.load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_doc = text_split.split_documents(question_gen)

    return final_doc


def llm_pipline(file_path:str):
    """

    Args:
        file_path (str): _description_
    """
    from langchain.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains import RetrievalQA
    from langchain_groq import ChatGroq
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from Interview-Question-Creator.src.prompt import *

    chunk_documents = file_process(file_path)

    groq_api_keys=os.getenv("GROQ_APIKEY")
    llm =ChatGroq(groq_api_key=groq_api_keys,model_name='Llama3-8b-8192')

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])
    
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    ques_gen_chain = load_summarize_chain(llm = llm, 
                                          chain_type = "refine",  
                                          verbose = True, 
                                          question_prompt=PROMPT_QUESTIONS, 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)
    ques = ques_gen_chain.run(chunk_documents)

    os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vdb=FAISS.from_documents(documents=documents_ques_gen ,embedding=embeddings)
    ques_list=ques.split("\n")

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=vdb.as_retriever())

    for question in ques_list:
    print("Question: ", question)
    answer = answer_generation_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\\n\\n")
    # Save answer to file
    with open("answers.txt", "a") as f:
        f.write("Question: " + question + "\\n")
        f.write("Answer: " + answer + "\\n")
        f.write("--------------------------------------------------\\n\\n")