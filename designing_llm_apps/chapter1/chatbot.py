# https://colab.research.google.com/github/corazzon/designing-llm-apps/blob/main/Chapter01/ChatwithyourPDF.ipynb#scrollTo=Fm8tMmNw1O9N

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import gradio as gr


# pdf 파일 파싱
loader = UnstructuredPDFLoader(input_file.name)
data = loader.load()

# 임베딩
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 벡터db
db = Chroma.from_documents(data, embeddings)

# 질의응답 예시
query = "환불은 어떻게 요청하나요?"
docs = db.similarity_search(query)

print(docs[0].page_content)

# 체인 정의
conversational_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.1),
    retriever=pdfsearch.as_retriever(search_kwargs={"k":3})
)

# 체인 호출
output = conversational_chain({
    'question': query,
    'chat_history': conversational_history
})

conversational_history += [(query, output['answer'])]

# UI 구축 (gradio)
with gr.Blocks as app:
    with gr.Row():
        chatbot = gr.Chatbot(value=[], elem_id='qa_chatbot').style(height=500)

    with gr.Row():
        with gr.Column(scale=0.80):
            textbox = gr.Textbox(
                placeholder="Enter text"
            ).style(container=False)

        with gr.Column(scale=0.10):
            upload_button = gr.UploadButton("Upload a PDF",
                                            file_types=[".pdf"]).style()
            

if __name__ == "__main__":
    app.launch()