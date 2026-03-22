// ============================================================
// File Uploader – Drop zone for SSE + Log files
// ============================================================

import { useCallback } from 'react';
import { useStore } from '../../store/useStore';
import { Upload, FileText, Zap } from 'lucide-react';

export function FileUploader() {
  const loadSSE = useStore(s => s.loadSSE);
  const loadLog = useStore(s => s.loadLog);
  const sseFileName = useStore(s => s.sseFileName);
  const logFileName = useStore(s => s.logFileName);
  const isLoading = useStore(s => s.isLoading);

  const handleFile = useCallback(
    (file: File, type: 'sse' | 'log') => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (type === 'sse') {
          loadSSE(text, file.name);
        } else {
          loadLog(text, file.name);
        }
      };
      reader.readAsText(file);
    },
    [loadSSE, loadLog]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent, type: 'sse' | 'log') => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file, type);
    },
    [handleFile]
  );

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>, type: 'sse' | 'log') => {
      const file = e.target.files?.[0];
      if (file) handleFile(file, type);
    },
    [handleFile]
  );

  return (
    <div className="flex gap-3">
      {/* SSE Upload */}
      <label
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all text-xs ${
          sseFileName
            ? 'bg-emerald-900/50 border border-emerald-700 text-emerald-300'
            : 'bg-gray-800 border border-gray-600 text-gray-300 hover:border-blue-500'
        }`}
        onDrop={(e) => handleDrop(e, 'sse')}
        onDragOver={(e) => e.preventDefault()}
      >
        {sseFileName ? (
          <>
            <Zap className="w-3.5 h-3.5" />
            <span className="max-w-[120px] truncate">{sseFileName}</span>
          </>
        ) : (
          <>
            <Upload className="w-3.5 h-3.5" />
            <span>SSE Stream</span>
          </>
        )}
        <input
          type="file"
          className="hidden"
          accept=".txt,.log,.sse"
          onChange={(e) => handleInput(e, 'sse')}
        />
      </label>

      {/* Log Upload */}
      <label
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all text-xs ${
          logFileName
            ? 'bg-emerald-900/50 border border-emerald-700 text-emerald-300'
            : 'bg-gray-800 border border-gray-600 text-gray-300 hover:border-blue-500'
        }`}
        onDrop={(e) => handleDrop(e, 'log')}
        onDragOver={(e) => e.preventDefault()}
      >
        {logFileName ? (
          <>
            <FileText className="w-3.5 h-3.5" />
            <span className="max-w-[120px] truncate">{logFileName}</span>
          </>
        ) : (
          <>
            <Upload className="w-3.5 h-3.5" />
            <span>Proxy Log</span>
          </>
        )}
        <input
          type="file"
          className="hidden"
          accept=".txt,.log"
          onChange={(e) => handleInput(e, 'log')}
        />
      </label>

      {isLoading && (
        <div className="flex items-center gap-1.5 text-xs text-blue-400">
          <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          Parsing...
        </div>
      )}
    </div>
  );
}
