/**
 * Sidebar Component
 * Shows chat sessions and navigation
 */

import React from 'react';
import { Plus, Trash2, MessageCircle } from 'lucide-react';
import type { ChatSession } from '@/types';
import { formatDate, isToday, formatTime } from '@utils/index';
import './Sidebar.css';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  loading?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  currentSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  loading = false,
}) => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>CHAT HISTORY</h1>
        <button
          onClick={onCreateSession}
          className="new-chat-btn"
          disabled={loading}
          title="New chat"
        >
          <Plus size={18} />
        </button>
      </div>

      <div className="sessions-list">
        {sessions.length === 0 ? (
          <div className="empty-sessions">
            <MessageCircle size={32} />
            <p>No conversations yet</p>
          </div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${currentSessionId === session.id ? 'active' : ''}`}
              onClick={() => onSelectSession(session.id)}
            >
              <div className="session-info">
                <div className="session-title">{session.title}</div>
                <div className="session-meta">
                  {isToday(session.created_at)
                    ? formatTime(session.created_at)
                    : formatDate(session.created_at)}
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(session.id);
                }}
                className="delete-btn"
                title="Delete session"
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Sidebar;
