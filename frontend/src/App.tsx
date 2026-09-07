/**
 * Main App Component
 * Root component orchestrating the entire application
 */

import React, { useEffect } from 'react';
import { useChat, useSessions } from '@hooks/index';
import Chat from '@components/Chat/Chat';
import Sidebar from '@components/Sidebar/Sidebar';
import SourcesPanel from '@components/SourcesPanel/SourcesPanel';
import './App.css';

const App: React.FC = () => {
  const {
    sessionId,
    messages,
    loading,
    error,
    sendMessage,
    createNewSession,
    deleteCurrentSession,
  } = useChat();

  const { sessions, loading: sessionsLoading, refreshSessions } = useSessions();

  // Initialize with a new session on mount
  useEffect(() => {
    if (!sessionId && sessions.length === 0) {
      createNewSession();
    }
  }, []);

  const handleSelectSession = (id: string) => {
    // This would be handled by re-initializing the chat with the session ID
    window.location.hash = `#session/${id}`;
    window.location.reload();
  };

  const handleDeleteSession = async (_id: string) => {
    await deleteCurrentSession();
    await refreshSessions();
  };

  // Get last message's sources for the panel
  const lastAssistantMessage = messages.filter((m) => m.role === 'assistant').pop();
  const sources = lastAssistantMessage?.sources;

  return (
    <div className="app">
      <div className="main-layout">
        <Sidebar
          sessions={sessions}
          currentSessionId={sessionId}
          onSelectSession={handleSelectSession}
          onCreateSession={createNewSession}
          onDeleteSession={handleDeleteSession}
          loading={sessionsLoading}
        />

        <div className="chat-area">
          <Chat
            messages={messages}
            loading={loading}
            onSendMessage={sendMessage}
          />
        </div>

        {sources && sources.length > 0 && (
          <SourcesPanel sources={sources} />
        )}
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default App;
