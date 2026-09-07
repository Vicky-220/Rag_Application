/**
 * API Service for communicating with the backend
 * Handles all HTTP requests and WebSocket connections
 */

import axios, { AxiosInstance } from 'axios';
import type {
  ChatSession,
  ChatMessageResponse,
  KnowledgeStats,
  KnowledgeStructure,
  UploadResponse,
  Message,
} from '@/types';

class APIService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = '/api') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: this.baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // ============ Chat Endpoints ============

  /**
   * Send a message and get a response
   */
  async sendMessage(message: string, sessionId?: string): Promise<ChatMessageResponse> {
    const response = await this.client.post<ChatMessageResponse>('/chat/message', {
      message,
      session_id: sessionId,
    });
    return response.data;
  }

  /**
   * Get WebSocket URL for streaming
   */
  getWebSocketURL(sessionId: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/api/chat/ws/${sessionId}`;
  }

  // ============ Session Endpoints ============

  /**
   * Get all chat sessions
   */
  async getSessions(): Promise<ChatSession[]> {
    const response = await this.client.get<{ sessions: ChatSession[]; count: number }>('/chat/sessions');
    return response.data.sessions;
  }

  /**
   * Get a specific session with its messages
   */
  async getSession(sessionId: string): Promise<{ session: ChatSession; messages: Message[] }> {
    const response = await this.client.get(`/chat/session/${sessionId}`);
    return response.data;
  }

  /**
   * Create a new chat session
   */
  async createSession(title?: string): Promise<{ session_id: string; title: string; status: string }> {
    const response = await this.client.post('/chat/session/create', null, {
      params: { title },
    });
    return response.data;
  }

  /**
   * Delete a chat session
   */
  async deleteSession(sessionId: string): Promise<{ status: string; session_id: string }> {
    const response = await this.client.delete(`/chat/session/${sessionId}`);
    return response.data;
  }

  // ============ Knowledge Base Endpoints ============

  /**
   * Get knowledge base statistics
   */
  async getKnowledgeStats(): Promise<{
    vector_db: KnowledgeStats;
    pdf_files: string[];
    total_pdfs: number;
    knowledge_base_path: string;
  }> {
    const response = await this.client.get('/knowledge/stats');
    return response.data;
  }

  /**
   * Get knowledge base structure
   */
  async getKnowledgeStructure(): Promise<{ structure: KnowledgeStructure }> {
    const response = await this.client.get('/knowledge/structure');
    return response.data;
  }

  /**
   * Get all chunks
   */
  async getAllChunks(limit?: number): Promise<{
    total_chunks: number;
    returned: number;
    chunks: any[];
  }> {
    const response = await this.client.get('/knowledge/chunks', {
      params: { limit },
    });
    return response.data;
  }

  /**
   * Refresh knowledge base
   */
  async refreshKnowledge(): Promise<{
    status: string;
    total_documents: number;
    total_chunks: number;
    new_chunks_added: number;
  }> {
    const response = await this.client.post('/knowledge/refresh');
    return response.data;
  }

  /**
   * Search knowledge base
   */
  async searchKnowledge(query: string, k?: number): Promise<{
    query: string;
    results_count: number;
    results: any[];
  }> {
    const response = await this.client.get('/knowledge/search', {
      params: { query, k },
    });
    return response.data;
  }

  // ============ Upload Endpoints ============

  /**
   * Upload a PDF file
   */
  async uploadPDF(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<UploadResponse>('/chat/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // ============ Health Check ============

  /**
   * Check backend health
   */
  async healthCheck(): Promise<{ status: string; service: string }> {
    const response = await this.client.get('/');
    return response.data;
  }
}

// Export singleton instance
export default new APIService();
