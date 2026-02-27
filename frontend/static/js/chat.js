/**
 * Chat Logic — Local AI Coding Assistant
 * =======================================
 * Handles user input, sends requests to the Flask backend,
 * renders AI responses with syntax highlighting, and
 * manages the chat UI state.
 */

// ── DOM Elements ────────────────────────────────────────────────────
const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const chatMessages = document.getElementById("chatMessages");

// ── Auto-resize Textarea ────────────────────────────────────────────
promptInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 150) + "px";
});

// ── Enter to Send (Shift+Enter for new line) ────────────────────────
promptInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

// ── Form Submission ─────────────────────────────────────────────────
chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Add the user's message to the chat
    addMessage(prompt, "user");

    // Clear input and reset height
    promptInput.value = "";
    promptInput.style.height = "auto";

    // Disable input while generating
    setLoading(true);

    // Show loading indicator
    const loadingEl = addLoadingMessage();

    try {
        // Send request to Flask backend
        const response = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                max_length: 256,
                temperature: 0.7,
            }),
        });

        const data = await response.json();

        // Remove loading indicator
        loadingEl.remove();

        if (data.error) {
            addErrorMessage(data.error);
        } else {
            addMessage(data.generated_code, "ai");
        }
    } catch (error) {
        loadingEl.remove();
        addErrorMessage(
            "Failed to connect to the server. Please check that both " +
            "the FastAPI backend and Flask frontend are running."
        );
    } finally {
        setLoading(false);
    }
});

// ── Add a Chat Message ──────────────────────────────────────────────
/**
 * Create and append a message bubble to the chat.
 * @param {string} text - The message text (may contain code).
 * @param {string} role - Either "user" or "ai".
 */
function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}-message`;

    const avatar = role === "ai" ? "🤖" : "👤";
    const name = role === "ai" ? "AI Assistant" : "You";

    // Format code blocks (detect ``` fences or treat as code if AI)
    let contentHTML;
    if (role === "ai") {
        contentHTML = formatCodeResponse(text);
    } else {
        contentHTML = `<p>${escapeHTML(text)}</p>`;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">${name}</div>
            ${contentHTML}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Highlight any code blocks
    messageDiv.querySelectorAll("pre code").forEach((block) => {
        hljs.highlightElement(block);
    });

    // Add copy buttons to code blocks
    messageDiv.querySelectorAll("pre").forEach(addCopyButton);
}

// ── Format AI Response ──────────────────────────────────────────────
/**
 * Wrap the AI's response in a code block with syntax highlighting.
 * If the response contains markdown-style ```python fences, parse them.
 * Otherwise, treat the entire response as code.
 * @param {string} text - Raw AI response text.
 * @returns {string} HTML string.
 */
function formatCodeResponse(text) {
    // Check if the response contains explicit code fences
    const fenceRegex = /```(\w+)?\n([\s\S]*?)```/g;
    let match;
    let result = "";
    let lastIndex = 0;
    let hasCodeFence = false;

    while ((match = fenceRegex.exec(text)) !== null) {
        hasCodeFence = true;
        const lang = match[1] || "python";
        const code = match[2].trim();

        // Add any text before this code block
        const before = text.slice(lastIndex, match.index).trim();
        if (before) {
            result += `<p>${escapeHTML(before)}</p>`;
        }

        result += `<pre><code class="language-${lang}">${escapeHTML(code)}</code></pre>`;
        lastIndex = match.index + match[0].length;
    }

    if (hasCodeFence) {
        // Add any remaining text after the last code block
        const remaining = text.slice(lastIndex).trim();
        if (remaining) {
            result += `<p>${escapeHTML(remaining)}</p>`;
        }
        return result;
    }

    // No fences found — treat entire response as Python code
    return `<pre><code class="language-python">${escapeHTML(text)}</code></pre>`;
}

// ── Copy Button for Code Blocks ─────────────────────────────────────
/**
 * Add a "Copy" button to a <pre> element.
 * @param {HTMLElement} preEl - The <pre> element.
 */
function addCopyButton(preEl) {
    preEl.style.position = "relative";
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", () => {
        const code = preEl.querySelector("code").textContent;
        navigator.clipboard.writeText(code).then(() => {
            btn.textContent = "Copied!";
            setTimeout(() => (btn.textContent = "Copy"), 2000);
        });
    });
    preEl.appendChild(btn);
}

// ── Loading Indicator ───────────────────────────────────────────────
/**
 * Add a loading ("thinking") animation to the chat.
 * @returns {HTMLElement} The loading element (so it can be removed later).
 */
function addLoadingMessage() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message ai-message";
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-header">AI Assistant</div>
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

// ── Error Message ───────────────────────────────────────────────────
/**
 * Show an error message in the chat.
 * @param {string} errorText - The error description.
 */
function addErrorMessage(errorText) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message ai-message";
    messageDiv.innerHTML = `
        <div class="message-avatar">⚠️</div>
        <div class="message-content">
            <div class="error-text">${escapeHTML(errorText)}</div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ── Suggestion Chips ────────────────────────────────────────────────
/**
 * Fill the input with a clicked suggestion.
 * @param {HTMLElement} el - The clicked <li> element.
 */
function useSuggestion(el) {
    promptInput.value = el.textContent;
    promptInput.focus();
    promptInput.style.height = "auto";
    promptInput.style.height = promptInput.scrollHeight + "px";
}

// ── Utility Functions ───────────────────────────────────────────────

/** Scroll the chat window to the latest message. */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/** Toggle the send button and input state. */
function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    promptInput.disabled = isLoading;
    if (!isLoading) promptInput.focus();
}

/** Escape HTML special characters to prevent XSS. */
function escapeHTML(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
