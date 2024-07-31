import os
import time
import streamlit as st
from PIL import Image
from io import BytesIO
from gemini_response import generate_gemini_response
from chat_history import clear_chat_history, initialize_chat, display_chat_messages
from multimodal import ImageQuestionAnswering
from img_link import process_user_input  # Importing the function

# Set up Streamlit app
st.set_page_config(page_title="🦙💬 Receipts Chatbot 📖")

# Sidebar configuration
with st.sidebar:
    st.title('🦙💬 Receipts Chatbot 📖')
    st.button('Clear Chat History', on_click=clear_chat_history)

# Cache the model using st.cache_resource
@st.cache_resource
def load_image_qa_model():
    return ImageQuestionAnswering()

# Initialize chat
initialize_chat()

# Display chat messages
display_chat_messages()

# Main chat application
st.title("Image Question Answering")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded previously in the session
# if 'uploaded_image' not in st.session_state:
#     img = None
img = None
# If a new file is uploaded, store it in the session state
if uploaded_file:
    img = Image.open(uploaded_file)
    print(type(Image.open(uploaded_file)))
    st.image(img, caption='Uploaded Image', use_column_width=True)

# Text input for chat
if prompt := st.chat_input():
    # Process the user input to separate text and image (if any)
    user_prompt, image = process_user_input(prompt)
    
    # Store the processed user input and image in session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    if image:
        img = image
        print(type(img))
        st.image(img, caption='Uploaded Image from URL', use_column_width=True)
        
    if user_prompt:
        if img:
            st.session_state.messages.append({"role": "user", "content": img, "type": "image"})
            st.session_state.messages.append({"role": "user", "content": user_prompt})
        else:
            st.session_state.messages.append({"role": "user", "content": user_prompt})

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        # Load the model using cache
        image_qa_instance = load_image_qa_model()
        result = None
        if img:
            print("using image")
            result = image_qa_instance.generate_response(img, user_prompt)
        else:
            print("do not include image")
            result = image_qa_instance.generate_response(None, user_prompt)

        if result:
            placeholder = st.empty()

            def stream_data():
                for word in result.split(" "):
                    yield word + " "
                    time.sleep(0.04)
            placeholder.write("".join([word for word in stream_data()]))

        message = {"role": "assistant", "content": result}
        st.session_state.messages.append(message)
