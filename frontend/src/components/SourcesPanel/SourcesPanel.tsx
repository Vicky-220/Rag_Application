/**
 * Sources Panel Component
 * Displays retrieved sources/chunks for the current response
 */

import React from 'react';
import { FileText, Copy } from 'lucide-react';
import type { ChunkSource } from '@/types';
import { copyToClipboard } from '@utils/index';
import './SourcesPanel.css';

interface SourcesPanelProps {
  sources: ChunkSource[] | undefined;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({ sources }) => {
  const [copiedId, setCopiedId] = React.useState<string | null>(null);

  const handleCopy = async (content: string, id: string) => {
    const success = await copyToClipboard(content);
    if (success) {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-panel">
      <div className="sources-header">
        <h3>
          <FileText size={16} />
          Sources ({sources.length})
        </h3>
      </div>

      <div className="sources-list">
        {sources.map((source) => (
          <div key={source.id} className="source-item">
            <div className="source-header">
              <div className="source-meta">
                <span className="source-file">{source.source}</span>
                <span className="source-page">Page {source.page}</span>
                <span className="source-score">Score: {source.score.toFixed(2)}</span>
              </div>
              <div className="source-actions">
                <button
                  onClick={() => handleCopy(source.content, source.id)}
                  className="action-btn"
                  title="Copy"
                >
                  <Copy size={14} />
                  {copiedId === source.id && <span className="copied">Copied!</span>}
                </button>
              </div>
            </div>

            <div className="source-content">
              {source.content.length > 300
                ? source.content.substring(0, 300) + '...'
                : source.content}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SourcesPanel;
