/**
 * Main Chat Component
 * Displays chat messages and handles message input
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader } from 'lucide-react';
import type { Message } from '@/types';
import './Chat.css';

interface ChatProps {
  messages: Message[];
  loading: boolean;
  onSendMessage: (message: string) => void;
}

export const Chat: React.FC<ChatProps> = ({ messages, loading, onSendMessage }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    onSendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Welcome to RAG Chat</h2>
            <p>Ask me anything about your knowledge base</p>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <div key={index} className={`message message-${message.role}`}>
                <div className="message-bubble">
                  <div className="message-content">{message.content}</div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <small>Sources: {message.sources.length} chunk(s)</small>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message message-assistant">
                <div className="message-bubble">
                  <Loader size={16} className="spinner" /> Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className="message-input"
        />
        <button type="submit" disabled={loading || !input.trim()} className="send-button">
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};

export default Chat;
