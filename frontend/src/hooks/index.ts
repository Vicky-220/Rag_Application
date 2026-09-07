/**
 * Custom React hooks for the RAG application
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { Message, ChatSession } from '@/types';
import apiService from '@services/apiService';

/**
 * Hook for managing chat state and operations
 */
export const useChat = (initialSessionId?: string) => {
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId || null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load session messages
  useEffect(() => {
    if (sessionId) {
      loadSessionMessages(sessionId);
    }
  }, [sessionId]);

  const loadSessionMessages = useCallback(async (id: string) => {
    try {
      setLoading(true);
      const data = await apiService.getSession(id);
      setMessages(data.messages);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session');
    } finally {
      setLoading(false);
    }
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!sessionId && !content.trim()) return;

      try {
        setLoading(true);
        setError(null);

        // Add user message optimistically
        const userMessage: Message = {
          role: 'user',
          content,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);

        // Send to API
        const response = await apiService.sendMessage(content, sessionId || undefined);

        // Update session ID if new
        if (!sessionId) {
          setSessionId(response.session_id);
        }

        // Add assistant message
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.response,
          sources: response.sources,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to send message');
        // Remove optimistic user message on error
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  const createNewSession = useCallback(async (title?: string) => {
    try {
      const response = await apiService.createSession(title);
      setSessionId(response.session_id);
      setMessages([]);
      setError(null);
      return response.session_id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
      return null;
    }
  }, []);

  const deleteCurrentSession = useCallback(async () => {
    if (!sessionId) return false;

    try {
      await apiService.deleteSession(sessionId);
      setSessionId(null);
      setMessages([]);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete session');
      return false;
    }
  }, [sessionId]);

  return {
    sessionId,
    messages,
    loading,
    error,
    sendMessage,
    createNewSession,
    deleteCurrentSession,
  };
};

/**
 * Hook for managing sessions
 */
export const useSessions = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiService.getSessions();
      setSessions(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
    // Refresh sessions every 30 seconds
    const interval = setInterval(loadSessions, 30000);
    return () => clearInterval(interval);
  }, [loadSessions]);

  return {
    sessions,
    loading,
    error,
    refreshSessions: loadSessions,
  };
};

/**
 * Hook for knowledge base operations
 */
export const useKnowledgeBase = () => {
  const [stats, setStats] = useState<any>(null);
  const [structure, setStructure] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiService.getKnowledgeStats();
      setStats(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadStructure = useCallback(async () => {
    try {
      const data = await apiService.getKnowledgeStructure();
      setStructure(data.structure);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load structure');
    }
  }, []);

  const searchKnowledge = useCallback(
    async (query: string, k?: number) => {
      try {
        return await apiService.searchKnowledge(query, k);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Search failed');
        throw err;
      }
    },
    []
  );

  const refreshKnowledge = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiService.refreshKnowledge();
      await loadStats();
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Refresh failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadStats]);

  return {
    stats,
    structure,
    loading,
    error,
    loadStats,
    loadStructure,
    searchKnowledge,
    refreshKnowledge,
  };
};

/**
 * Hook for managing WebSocket connections
 */
export const useWebSocket = (url: string) => {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);

    return () => {
      ws.current?.close();
    };
  }, [url]);

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    }
  }, []);

  return {
    ws: ws.current,
    connected,
    send,
  };
};
