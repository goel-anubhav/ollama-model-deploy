# streamlit_frontend.py
import streamlit as st
import requests
import json
import time
from datetime import datetime
import sseclient  # You'll need to install this: pip install sseclient-py

# === Configuration ===
API_BASE_URL = "http://34.83.223.15:8000"

# === Page Configuration ===
st.set_page_config(
    page_title="Ollama Chat Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# === Session State Initialization ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# === Helper Functions ===
def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status code: {response.status_code}"}
    except requests.RequestException as e:
        return False, {"error": str(e)}

def send_non_streaming_request(question, include_context=False, context_limit=5):
    """Send non-streaming request to API"""
    try:
        payload = {
            "question": question,
            "stream": False,
            "include_context": include_context,
            "context_limit": context_limit
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, {"error": str(e)}

def send_streaming_request(question, include_context=False, context_limit=5):
    """Send streaming request to API"""
    try:
        payload = {
            "question": question,
            "stream": True,
            "include_context": include_context,
            "context_limit": context_limit
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        return True, response
    except requests.RequestException as e:
        return False, {"error": str(e)}

def get_chat_history():
    """Get chat history from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/chat-history", timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, {"error": str(e)}

def get_formatted_chat_history(limit=10):
    """Get formatted chat history from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/chat-history/formatted?limit={limit}", timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, {"error": str(e)}

def clear_chat_history():
    """Clear chat history on server"""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat-history", timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, {"error": str(e)}

# === Main App ===
def main():
    st.title("🤖 Ollama Chat Interface")
    st.markdown("---")
    
    # === Sidebar ===
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Health Check
        st.subheader("🔍 API Status")
        if st.button("Check API Health"):
            is_healthy, health_data = check_api_health()
            if is_healthy:
                st.markdown(f'<div class="status-box status-success">✅ API is healthy<br>Model: {health_data.get("model", "Unknown")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box status-error">❌ API is not responding<br>Error: {health_data.get("error", "Unknown")}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat Settings
        st.subheader("💬 Chat Settings")
        stream_mode = st.checkbox("Enable Streaming", value=True, help="Get real-time responses")
        include_context = st.checkbox("Include Chat Context", value=False, help="Use previous conversations for context")
        
        if include_context:
            context_limit = st.slider("Context Conversations", min_value=1, max_value=10, value=5, 
                                    help="Number of previous conversations to include")
        else:
            context_limit = 5
        
        st.markdown("---")
        
        # Chat History Management
        st.subheader("📚 Chat History")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View History"):
                success, history_data = get_chat_history()
                if success:
                    st.text_area("Chat History", value=history_data.get("chat_history", "No history found"), height=200)
                else:
                    st.error(f"Failed to get history: {history_data.get('error')}")
        
        with col2:
            if st.button("Clear History"):
                success, result = clear_chat_history()
                if success:
                    st.success("History cleared!")
                    st.session_state.messages = []
                else:
                    st.error(f"Failed to clear history: {result.get('error')}")
        
        if st.button("Get Formatted History"):
            success, formatted_data = get_formatted_chat_history(context_limit)
            if success:
                st.text_area("Formatted History", value=formatted_data.get("formatted_history", "No history found"), height=200)
            else:
                st.error(f"Failed to get formatted history: {formatted_data.get('error')}")

    # === Main Chat Interface ===
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # === Chat Input ===
    st.markdown("---")
    
    # Create input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask a question:",
                placeholder="Type your question here...",
                height=100,
                key="user_question"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process user input
    if submit_button and user_input.strip() and not st.session_state.is_streaming:
        # Add user message to session state
        user_message = {
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.messages.append(user_message)
        
        # Show processing status
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        if stream_mode:
            # Streaming response
            st.session_state.is_streaming = True
            status_placeholder.markdown('<div class="status-box status-info">🔄 Streaming response...</div>', unsafe_allow_html=True)
            
            success, response = send_streaming_request(user_input.strip(), include_context, context_limit)
            
            if success:
                full_response = ""
                
                # Create a placeholder for the streaming response
                with response_placeholder.container():
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        <div id="streaming-response"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    streaming_text = st.empty()
                    
                    try:
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                                        if 'response' in data:
                                            full_response += data['response']
                                            streaming_text.markdown(f"**🤖 Assistant:** {full_response}")
                                        elif data.get('done', False):
                                            break
                                        elif 'error' in data:
                                            st.error(f"Streaming error: {data['error']}")
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        st.error(f"Streaming failed: {str(e)}")
                
                # Add assistant message to session state
                if full_response:
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.messages.append(assistant_message)
                    status_placeholder.markdown('<div class="status-box status-success">✅ Response completed!</div>', unsafe_allow_html=True)
                else:
                    status_placeholder.markdown('<div class="status-box status-error">❌ No response received</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f'<div class="status-box status-error">❌ Request failed: {response.get("error")}</div>', unsafe_allow_html=True)
            
            st.session_state.is_streaming = False
            
        else:
            # Non-streaming response
            status_placeholder.markdown('<div class="status-box status-info">🔄 Processing request...</div>', unsafe_allow_html=True)
            
            success, response_data = send_non_streaming_request(user_input.strip(), include_context, context_limit)
            
            if success:
                answer = response_data.get("answer", "No answer received")
                
                # Add assistant message to session state
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.messages.append(assistant_message)
                
                status_placeholder.markdown('<div class="status-box status-success">✅ Response received!</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f'<div class="status-box status-error">❌ Request failed: {response_data.get("error")}</div>', unsafe_allow_html=True)
        
        # Clear status after a few seconds
        time.sleep(2)
        status_placeholder.empty()
        
        # Rerun to update the chat display
        st.rerun()
    
    elif st.session_state.is_streaming:
        st.warning("⏳ Please wait for the current response to complete...")
    
    # === Footer ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Powered by Ollama API | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
