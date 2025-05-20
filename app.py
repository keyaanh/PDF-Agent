import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
import re
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from streamlit_chat import message
import importlib.util
from html import css, bot_template, user_template


def check_dependency(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def get_pdf_text_ocr(pdf_docs):
    text_data = []
    try:
        for pdf in pdf_docs:
            # Convert pdf -> images
            try:
                images = pdf2image.convert_from_bytes(pdf.read())
            except pdf2image.exceptions.PDFInfoNotInstalledError:
                st.error("Poppler is not installed or not in PATH. Please install Poppler and try again.")
                return []
            except Exception as e:
                st.error(f"Error converting PDF to images: {str(e)}")
                return []
            
            pdf_text = ""
            for page_num, image in enumerate(images):
                # Do OCR
                try:
                    text = pytesseract.image_to_string(image, config='--psm 6')
                    pdf_text += f"\n[Page {page_num + 1}]\n{text}"
                except Exception as e:
                    st.error(f"OCR failed for page {page_num + 1}: {str(e)}")
                    pdf_text += f"\n[Page {page_num + 1}]\n[OCR Failed]"
            text_data.append({"pdf_name": pdf.name, "text": pdf_text})
        return text_data
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return []


def clean_text(text):
    
    lines = text.split("\n")
    unique_lines = []
    seen = set()
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line and cleaned_line not in seen:
            unique_lines.append(cleaned_line)
            seen.add(cleaned_line)
    
    text = "\n".join(unique_lines)
    text = re.sub(r"PROOUCTS", "PRODUCTS", text, flags=re.IGNORECASE)
    text = re.sub(r"TUCSSON", "TUCSON", text, flags=re.IGNORECASE)
    text = re.sub(r"\(52E\)", "(520)", text)  # Fix phone number typo
    return text


def extract_structured_data(text):
    structured_data = {}
    
    patterns = {
        "delivery_ticket": r"DELIVERY TICKET\s*([A-Z0-9-]+)",
        "account": r"Account:\s*(\d+)",
        "branch": r"Branch:\s*([A-Z0-9]+)",
        "phone": r"Phone:\s*\(?(\d{3})[-.\s]*\d{3}[-.\s]*\d{4}\)?",
        "ship_to": r"Ship To:([\s\S]*?)(?=Bill To:|$)",
        "bill_to": r"Bill To:([\s\S]*?)(?=\n\n|$)",
        "po": r"PO:\s*([A-Z0-9\s]+)(?=\n|$)",  # Updated to handle spaces and multiline
        "order_date": r"Order Date:\s*(\d{2}/\d{2}/\d{2,4})"  # Allow 2 or 4 digit years
    }
    
    for key, pattern in patterns.items():
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                structured_data[key] = match.group(1).strip()
            else:
                structured_data[key] = "N/A"
                print(f"No match found for {key} with pattern {pattern}")
        except IndexError as e:
            print(f"IndexError for {key} with pattern {pattern}: {str(e)}")
            structured_data[key] = "N/A"
        except Exception as e:
            print(f"Error processing {key} with pattern {pattern}: {str(e)}")
            structured_data[key] = "N/A"
    
    # Extract table data
    table_data = []
    table_pattern = r"(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(.+)" 
    try:
        for match in re.finditer(table_pattern, text, re.MULTILINE):
            table_data.append({
                "ordered": match.group(1),
                "shipped": match.group(2),
                "unit": match.group(3),
                "item_code": match.group(4),
                "description": match.group(5).strip()
            })
    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
    
    structured_data["table"] = table_data
    return structured_data

def get_pdf_text(pdf_docs):
    raw_texts = get_pdf_text_ocr(pdf_docs)
    processed_texts = []
    for doc in raw_texts:
        cleaned_text = clean_text(doc["text"])
        structured_data = extract_structured_data(cleaned_text)
        
        text = f"PDF: {doc['pdf_name']}\n{cleaned_text}\nStructured Data: {structured_data}"
        processed_texts.append(text)
    return processed_texts

# chunking include metadata
def get_text_chunks(texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for text in texts:
        doc_chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {"source": text.split("\n")[0], "chunk_id": i}
            })
    return chunks

#  vector store 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [chunk["metadata"] for chunk in text_chunks]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vectorstore

# custom prompt
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant analyzing messy PDF documents, such as delivery tickets with OCR errors and inconsistent formatting. Use the provided text and structured data to answer questions accurately. If data is missing or ambiguous, provide the best possible answer based on context and indicate any uncertainties. Include relevant details like delivery ticket numbers, addresses, or item descriptions when applicable.
        
        Context: {context}
        Question: {question}
        Answer:
        """
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return conversation_chain

# Handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state['past'].append(user_question)
    st.session_state['generated'].append(response['answer'])
    st.session_state['history'].append((user_question, response['answer']))

# Main function
def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return

    if not check_dependency("streamlit_chat"):
        st.error("Missing required dependency 'streamlit_chat'. Please install it using 'pip install streamlit_chat'.")
        return

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Chat with Messy PDFs 📚</h1>
            <h3 style='color: white;'>Ask questions about your PDF documents</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(css, unsafe_allow_html=True)

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Upload PDFs and ask me anything about them 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    # Sidebar
    pdf_docs = st.sidebar.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

    # Process button
    if pdf_docs and st.sidebar.button("Process"):
        with st.spinner("Processing..."):
            
            raw_texts = get_pdf_text(pdf_docs)
            
            text_chunks = get_text_chunks(raw_texts)
            
            vectorstore = get_vectorstore(text_chunks)
            
            st.session_state.conversation = get_conversation_chain(vectorstore)

    # Containers for chat history 
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your PDF data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            if st.session_state.conversation is None:
                st.error("Please upload and process PDFs first.")
            else:
                with st.spinner("Processing..."):
                    handle_userinput(user_input)

    # Chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == '__main__':
    main()