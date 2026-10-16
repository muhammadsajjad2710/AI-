(() => {
  const STORAGE_KEY = "msaActiveConversationId";
  const transcript = document.getElementById("transcript");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const errorBanner = document.getElementById("error-banner");
  const statusPill = document.getElementById("status-pill");

  const EXAMPLES = [
    "Hello — what can you help me with?",
    "Explain how photosynthesis works in simple terms.",
    "What are the latest headlines about local AI models today?",
    "Who won the most recent FIFA World Cup?",
  ];

  let conversationId = null;
  let busy = false;

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value == null || value === false) return;
      if (key === "className") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else node.setAttribute(key, value);
    });
    children.forEach((child) => {
      if (child != null) node.appendChild(child);
    });
    return node;
  }

  function showError(message) {
    errorBanner.hidden = !message;
    errorBanner.textContent = message || "";
  }

  function setBusy(nextBusy, statusText) {
    busy = nextBusy;
    sendBtn.disabled = nextBusy;
    input.disabled = nextBusy;
    newChatBtn.disabled = nextBusy;
    sendBtn.textContent = nextBusy ? "Working" : "Send";
    document.querySelectorAll(".example-btn").forEach((button) => {
      button.disabled = nextBusy;
    });
    if (statusText) {
      const existing = document.getElementById("loading-row");
      if (existing) {
        const label = existing.querySelector(".loading-label");
        if (label) label.textContent = statusText;
      }
    }
  }

  function scrollToLatest() {
    const panel = document.querySelector(".chat-panel");
    panel.scrollTop = panel.scrollHeight;
  }

  function isSafeHttpUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }

  function renderEmptyState() {
    transcript.replaceChildren();
    const examples = EXAMPLES.map((question) =>
      el(
        "button",
        { type: "button", className: "example-btn" },
        [document.createTextNode(question)]
      )
    );
    examples.forEach((button, index) => {
      button.addEventListener("click", () => sendMessage(EXAMPLES[index]));
    });
    transcript.appendChild(
      el("div", { className: "empty-state" }, [
        el("h2", { text: "Ask a question" }),
        el("p", {
          text: "I answer ordinary questions directly. For current events or facts I do not know, I can search the web.",
        }),
        el("div", { className: "example-grid" }, examples),
      ])
    );
  }

  function appendUserMessage(text) {
    const empty = transcript.querySelector(".empty-state");
    if (empty) transcript.replaceChildren();
    transcript.appendChild(
      el("div", { className: "row user" }, [
        el("div", { className: "bubble", text }),
      ])
    );
    scrollToLatest();
  }

  function appendAssistantMessage(text, sources) {
    const sourceCards = (sources || [])
      .filter((source) => source && isSafeHttpUrl(source.url))
      .map((source) =>
        el(
          "a",
          {
            className: "source-card",
            href: source.url,
            target: "_blank",
            rel: "noopener noreferrer",
          },
          [
            el("strong", { text: source.title || source.url }),
            el("span", { text: new URL(source.url).hostname }),
          ]
        )
      );

    const stackChildren = [el("div", { className: "bubble", text })];
    if (sourceCards.length) {
      stackChildren.push(el("div", { className: "sources" }, sourceCards));
    }
    transcript.appendChild(
      el("div", { className: "row assistant" }, [
        el("div", { className: "assistant-stack" }, stackChildren),
      ])
    );
    scrollToLatest();
  }

  function showLoading() {
    const row = el("div", { className: "row assistant", id: "loading-row" }, [
      el("div", { className: "bubble loading" }, [
        el("span", { className: "dots" }, [
          el("span"),
          el("span"),
          el("span"),
        ]),
        el("span", { className: "loading-label", text: "Working on your answer" }),
      ]),
    ]);
    transcript.appendChild(row);
    window.setTimeout(() => {
      const label = document.querySelector("#loading-row .loading-label");
      if (label) label.textContent = "Searching the web if needed";
    }, 1200);
    scrollToLatest();
  }

  function clearLoading() {
    document.getElementById("loading-row")?.remove();
  }

  async function parseError(response) {
    try {
      const data = await response.json();
      if (data && data.error) return data.error;
    } catch {
      // Fall through to a generic message.
    }
    if (response.status === 404) return "That conversation was not found. Start a new chat.";
    if (response.status === 429) return "Search is rate-limited. Wait a moment and try again.";
    if (response.status === 503) return "The local model is unavailable. Check Ollama and try again.";
    if (response.status === 504) return "The assistant timed out. Try a shorter question.";
    return "The request failed. You can retry or start a new chat.";
  }

  async function createConversation() {
    const response = await fetch("/api/conversations", { method: "POST" });
    if (!response.ok) throw new Error(await parseError(response));
    const data = await response.json();
    conversationId = data.conversation_id;
    localStorage.setItem(STORAGE_KEY, conversationId);
    return conversationId;
  }

  async function loadMessages(id) {
    const response = await fetch(`/api/conversations/${id}/messages`);
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(await parseError(response));
    return response.json();
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/health");
      if (!response.ok) throw new Error("health failed");
      const data = await response.json();
      const ollama = data.ollama || {};
      statusPill.classList.remove("ok", "warn");
      statusPill.title = ollama.detail || "";
      if (data.status === "ok") {
        statusPill.classList.add("ok");
        statusPill.textContent = `Ready · ${ollama.model}`;
      } else if (!ollama.available) {
        statusPill.classList.add("warn");
        statusPill.textContent = "Ollama unavailable";
      } else if (!ollama.model_available) {
        statusPill.classList.add("warn");
        statusPill.textContent = `Missing model · ${ollama.model}`;
      } else if (ollama.supports_tools === false) {
        statusPill.classList.add("warn");
        statusPill.textContent = "Model cannot call tools";
      } else {
        statusPill.classList.add("warn");
        statusPill.textContent = "Local model unavailable";
      }
    } catch {
      statusPill.classList.add("warn");
      statusPill.title = "";
      statusPill.textContent = "Backend health check failed";
    }
  }

  async function startNewChat() {
    showError("");
    conversationId = await createConversation();
    renderEmptyState();
    input.focus();
  }

  async function restoreOrCreate() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      await startNewChat();
      return;
    }
    try {
      const payload = await loadMessages(saved);
      if (!payload) {
        await startNewChat();
        return;
      }
      conversationId = saved;
      const messages = payload.messages || [];
      if (!messages.length) {
        renderEmptyState();
        return;
      }
      transcript.replaceChildren();
      messages.forEach((message) => {
        if (message.role === "user") appendUserMessage(message.content);
        else appendAssistantMessage(message.content, message.sources || []);
      });
    } catch (error) {
      showError(error.message || "Could not restore the previous chat.");
      await startNewChat();
    }
  }

  async function sendMessage(rawText) {
    const text = (rawText || "").trim();
    if (!text || busy) return;
    showError("");
    if (!conversationId) await createConversation();
    appendUserMessage(text);
    input.value = "";
    setBusy(true);
    showLoading();
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: text,
        }),
      });
      if (!response.ok) throw new Error(await parseError(response));
      const data = await response.json();
      clearLoading();
      appendAssistantMessage(data.answer, data.sources || []);
    } catch (error) {
      clearLoading();
      showError(error.message || "Something went wrong. You can retry that question.");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(input.value);
    }
  });

  newChatBtn.addEventListener("click", () => {
    startNewChat().catch((error) => showError(error.message));
  });

  refreshHealth();
  restoreOrCreate().catch((error) => showError(error.message));
})();
