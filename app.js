/* ═══════════════════════════════════════════════════════════════════════════
   app.js — VoiceOps ZentroMart Support UI
   Handles: text chat, microphone recording, audio upload, status checks.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatMessages    = document.getElementById('chatMessages');
const messageInput    = document.getElementById('messageInput');
const sendBtn         = document.getElementById('sendBtn');
const recordBtn       = document.getElementById('recordBtn');
const audioUpload     = document.getElementById('audioUpload');
const recordingBadge  = document.getElementById('recordingBadge');
const typingIndicator = document.getElementById('typingIndicator');
const transcriptBar   = document.getElementById('transcriptBar');
const transcriptText  = document.getElementById('transcriptText');
const evidencePanel   = document.getElementById('evidencePanel');
const verifiedBadge   = document.getElementById('verifiedBadge');
const ticketId        = document.getElementById('ticketId');
const statusDot       = document.getElementById('statusDot');
const statusLabel     = document.getElementById('statusLabel');
const resetBtn        = document.getElementById('resetBtn');
const toastContainer  = document.getElementById('toastContainer');

// Quick topic chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => sendMessage(chip.dataset.msg));
});

let gnaniConfigured = false;
let browserVoiceActive = false;

// ── System status check ───────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    gnaniConfigured = Boolean(data.gnani_configured);
    if (data.gemini_configured) {
      statusDot.className = 'status-dot ok';
      statusLabel.textContent = gnaniConfigured
        ? `${data.rag_chunks} chunks · Gnani Voice`
        : `${data.rag_chunks} chunks · Browser Mic Live`;
    } else {
      statusDot.className = 'status-dot err';
      statusLabel.textContent = 'Gemini API key missing';
    }
  } catch {
    statusDot.className = 'status-dot err';
    statusLabel.textContent = 'Server unreachable';
  }
}
checkStatus();

// ── Auto-resize textarea ───────────────────────────────────────────────────────
messageInput.addEventListener('input', () => {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});
messageInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const msg = messageInput.value.trim();
    if (msg) sendMessage(msg);
  }
});
sendBtn.addEventListener('click', () => {
  const msg = messageInput.value.trim();
  if (msg) sendMessage(msg);
});

// ── Text chat ─────────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text) return;

  appendMessage(text, 'user');
  messageInput.value = '';
  messageInput.style.height = 'auto';
  showTyping(true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    showTyping(false);

    if (data.error) {
      appendMessage(`⚠️ Error: ${data.error}`, 'agent', true);
      toast(data.error, 'error');
    } else {
      appendMessage(data.reply, 'agent', false, data.ticket_ids);
      updateSidebar(data);
      if (browserVoiceActive) {
        speakText(data.reply);
        browserVoiceActive = false;
      }
    }
  } catch (err) {
    showTyping(false);
    appendMessage('Sorry, something went wrong. Please try again.', 'agent', true);
    toast('Network error — is the server running?', 'error');
  }
}

// ── Append message bubble ─────────────────────────────────────────────────────
function appendMessage(text, role, isError = false, ticketIds = []) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}-message`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'agent' ? 'Z' : 'U';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (isError) bubble.style.color = 'var(--red)';

  // Convert newlines to paragraphs
  text.split('\n').filter(Boolean).forEach(line => {
    const p = document.createElement('p');
    p.innerHTML = line
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>');
    bubble.appendChild(p);
  });

  // Ticket badges
  (ticketIds || []).forEach(tid => {
    if (tid) {
      const badge = document.createElement('div');
      badge.className = 'ticket-badge';
      badge.innerHTML = `✓ Ticket created: <code>${tid.slice(0, 16)}…</code>`;
      bubble.appendChild(badge);
    }
  });

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
}

// ── Typing indicator ─────────────────────────────────────────────────────────
function showTyping(show) {
  typingIndicator.classList.toggle('hidden', !show);
  if (show) chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Update sidebar info ───────────────────────────────────────────────────────
function updateSidebar(data) {
  // Verified badge
  if (data.verified) {
    verifiedBadge.textContent = 'Yes ✓';
    verifiedBadge.classList.add('verified');
  }

  // Ticket ID
  if (data.ticket_ids && data.ticket_ids.length > 0) {
    const tid = data.ticket_ids[0];
    ticketId.textContent = tid ? tid.slice(0, 14) + '…' : '—';
    toast('Ticket created successfully!', 'success');
  }

  // Evidence chips
  if (data.evidence && data.evidence.length > 0) {
    evidencePanel.innerHTML = '';
    data.evidence.forEach(ev => {
      const chip = document.createElement('div');
      chip.className = 'evidence-chip';
      chip.textContent = `[${ev.section_id}] ${ev.title}`;
      evidencePanel.appendChild(chip);
    });
  }
}

// ── Voice recording ────────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let speechRecognizer = null;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function speakText(text) {
  if ('speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
      // Clean markdown characters for spoken audio
      const clean = text
        .replace(/[*_#`~\[\]\(\)>]/g, '')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/\n+/g, ' ')
        .trim();
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.lang = 'en-IN';
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error:', e);
    }
  }
}

recordBtn.addEventListener('click', async () => {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
});

async function startRecording() {
  if (!gnaniConfigured && SpeechRecognition) {
    startBrowserSpeech();
    return;
  }
  await startGnaniMediaRecorder();
}

function startBrowserSpeech() {
  try {
    speechRecognizer = new SpeechRecognition();
    speechRecognizer.lang = 'en-IN';
    speechRecognizer.interimResults = true;
    speechRecognizer.continuous = false;

    transcriptBar.classList.remove('hidden');
    transcriptText.textContent = 'Listening… speak into your microphone';

    isRecording = true;
    browserVoiceActive = true;
    recordBtn.classList.add('recording');
    recordingBadge.classList.remove('hidden');

    speechRecognizer.onresult = (event) => {
      let interim = '';
      let final = '';
      for (let i = 0; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      const text = final || interim;
      transcriptText.textContent = text;
      messageInput.value = text;
    };

    speechRecognizer.onerror = (event) => {
      console.warn('Speech recognition error:', event.error);
      stopRecording();
      if (event.error === 'not-allowed') {
        toast('Microphone access blocked. Click the lock/tune icon in the browser address bar to allow.', 'error');
      } else if (event.error !== 'no-speech') {
        toast('Speech error: ' + event.error, 'error');
      }
    };

    speechRecognizer.onend = () => {
      stopRecording();
      const text = messageInput.value.trim();
      if (text) {
        sendMessage(text);
      }
    };

    speechRecognizer.start();
    toast('Listening… speak your question.', 'info');
  } catch (err) {
    stopRecording();
    toast('Could not start speech recognition: ' + err.message, 'error');
  }
}

async function startGnaniMediaRecorder() {
  if (!gnaniConfigured) {
    toast('GNANI_API_KEY is not set in .env. Please configure your Gnani account.', 'error');
    transcriptBar.classList.remove('hidden');
    transcriptText.textContent = 'GNANI_API_KEY missing. Add it to .env or use Google Chrome/Edge for browser speech.';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks  = [];

    let mimeType = '';
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      mimeType = 'audio/webm;codecs=opus';
    } else if (MediaRecorder.isTypeSupported('audio/webm')) {
      mimeType = 'audio/webm';
    } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
      mimeType = 'audio/mp4';
    }

    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const type = mimeType || 'audio/webm';
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(audioChunks, { type });
      submitAudio(blob, `recording.${ext}`);
      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    browserVoiceActive = false;
    recordBtn.classList.add('recording');
    recordingBadge.classList.remove('hidden');
    toast('Recording… click the mic again when done speaking.', 'info');
  } catch (err) {
    toast('Microphone access denied: ' + err.message, 'error');
  }
}

function stopRecording() {
  if (speechRecognizer) {
    try { speechRecognizer.stop(); } catch {}
    speechRecognizer = null;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch {}
  }
  isRecording = false;
  recordBtn.classList.remove('recording');
  recordingBadge.classList.add('hidden');
}

// ── Audio file upload ─────────────────────────────────────────────────────────
audioUpload.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) submitAudio(file, file.name);
  audioUpload.value = '';
});

// ── Submit audio to voice API ─────────────────────────────────────────────────
async function submitAudio(blob, filename) {
  showTyping(true);
  transcriptBar.classList.remove('hidden');
  transcriptText.textContent = 'Processing audio…';

  const formData = new FormData();
  formData.append('audio', blob, filename);

  try {
    const res  = await fetch('/api/voice', { method: 'POST', body: formData });
    const data = await res.json();
    showTyping(false);

    if (data.error) {
      transcriptText.textContent = data.error;
      toast(data.error, 'error');
      return;
    }

    // Show transcript
    if (data.transcript) {
      transcriptText.textContent = data.transcript;
      appendMessage(data.transcript, 'user');
    }

    // Show agent reply
    const msgWrapper = appendMessage(data.reply, 'agent', false, data.ticket_ids);

    // Attach audio player if TTS audio is available
    if (data.audio_url) {
      const playerDiv = document.createElement('div');
      playerDiv.className = 'audio-player';
      const audio = document.createElement('audio');
      audio.src = data.audio_url;
      audio.controls = true;
      audio.autoplay = true;
      playerDiv.appendChild(audio);
      msgWrapper.querySelector('.bubble').appendChild(playerDiv);
    }

    updateSidebar(data);
  } catch (err) {
    showTyping(false);
    toast('Voice processing failed: ' + err.message, 'error');
  }
}

// ── Reset conversation ────────────────────────────────────────────────────────
resetBtn.addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  chatMessages.innerHTML = `
    <div class="message agent-message">
      <div class="avatar">Z</div>
      <div class="bubble">
        <p>Hi! I'm <strong>Zara</strong>, your ZentroMart support assistant.</p>
        <p>I can help with orders, returns, refunds, deliveries, and more. How can I help you today?</p>
      </div>
    </div>`;
  evidencePanel.innerHTML = '<p class="empty-hint">Relevant policy sections appear here.</p>';
  verifiedBadge.textContent = 'No';
  verifiedBadge.classList.remove('verified');
  ticketId.textContent = '—';
  transcriptBar.classList.add('hidden');
  toast('Conversation reset.', 'info');
});

// ── Toast helper ──────────────────────────────────────────────────────────────
function toast(message, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ';
  el.innerHTML = `<span>${icon}</span> ${message}`;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
