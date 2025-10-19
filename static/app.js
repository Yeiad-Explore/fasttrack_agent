// Fast Track AI Agent - Frontend JavaScript

// State
let sessionId = generateSessionId();
let isProcessing = false;

// DOM Elements
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const clearChatBtn = document.getElementById('clearChatBtn');
const statsBtn = document.getElementById('statsBtn');
const indexKbBtn = document.getElementById('indexKbBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const toast = document.getElementById('toast');

// Utility Functions
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showLoading(message = 'Processing...') {
    loadingText.textContent = message;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    // Simple markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

// Message Functions
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">You</div>
        <div class="message-content">
            <p>${formatMessage(text)}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addAssistantMessage(text, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="message-sources">
                <h4>Sources:</h4>
                ${sources.map(s => `
                    <div class="source-item">
                        ${s.source} (Relevance: ${(s.relevance_score * 100).toFixed(1)}%)
                    </div>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <p>${formatMessage(text)}</p>
            ${sourcesHtml}
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant-message';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// API Functions
async function sendQuery(query) {
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId,
                stream: false,
                top_k: 5
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending query:', error);
        throw error;
    }
}

async function indexKnowledgeBase() {
    try {
        showLoading('Indexing knowledge base... This may take a few minutes.');

        const response = await fetch('/api/index-kb', {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to index knowledge base');
        }

        const data = await response.json();
        hideLoading();

        showToast(`Indexed ${data.total_files} files with ${data.total_chunks} chunks`, 'success');
        loadStats(); // Refresh stats
    } catch (error) {
        hideLoading();
        console.error('Error indexing KB:', error);
        showToast('Failed to index knowledge base', 'error');
    }
}

async function uploadDocument(file) {
    try {
        showLoading('Uploading and indexing document...');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to upload document');
        }

        const data = await response.json();
        hideLoading();

        showToast(`${data.filename} uploaded successfully (${data.chunks_created} chunks)`, 'success');
        loadStats(); // Refresh stats
    } catch (error) {
        hideLoading();
        console.error('Error uploading document:', error);
        showToast('Failed to upload document', 'error');
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');

        if (!response.ok) {
            throw new Error('Failed to load stats');
        }

        const data = await response.json();

        document.getElementById('totalDocs').textContent = data.total_documents;
        document.getElementById('indexSize').textContent = data.index_size;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function clearConversationHistory() {
    try {
        const response = await fetch(`/api/clear-history/${sessionId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to clear history');
        }

        // Clear chat UI
        chatMessages.innerHTML = `
            <div class="message assistant-message">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <p>Chat history cleared. How can I help you today?</p>
                </div>
            </div>
        `;

        // Generate new session ID
        sessionId = generateSessionId();

        showToast('Chat cleared', 'success');
    } catch (error) {
        console.error('Error clearing history:', error);
        showToast('Failed to clear chat', 'error');
    }
}

// Event Handlers
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = messageInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;

    // Add user message
    addUserMessage(query);
    messageInput.value = '';

    // Show typing indicator
    addTypingIndicator();

    try {
        // Send query
        const response = await sendQuery(query);

        // Remove typing indicator
        removeTypingIndicator();

        // Add assistant message
        addAssistantMessage(response.answer, response.sources);
    } catch (error) {
        removeTypingIndicator();
        addAssistantMessage('Sorry, I encountered an error processing your request. Please try again.');
        showToast('Error processing query', 'error');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
});

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Allow Enter to send, Shift+Enter for new line
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

clearChatBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
        clearConversationHistory();
    }
});

statsBtn.addEventListener('click', () => {
    loadStats();
    showToast('Stats refreshed', 'success');
});

indexKbBtn.addEventListener('click', () => {
    if (confirm('This will index all PDF files in the kb/ folder. Continue?')) {
        indexKnowledgeBase();
    }
});

uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        if (!file.name.endsWith('.pdf')) {
            showToast('Please select a PDF file', 'error');
            return;
        }
        uploadDocument(file);
        fileInput.value = ''; // Reset input
    }
});

// Quick questions
document.querySelectorAll('.quick-question').forEach(btn => {
    btn.addEventListener('click', () => {
        const question = btn.textContent;
        messageInput.value = question;
        messageInput.focus();
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    messageInput.focus();

    // Check health
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            console.log('API Status:', data);
        })
        .catch(err => {
            console.error('API health check failed:', err);
            showToast('Warning: Could not connect to API', 'warning');
        });
});
