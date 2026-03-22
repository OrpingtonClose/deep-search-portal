// ============================================================
// Header – Top bar with title, file uploaders, nav
// ============================================================

import { FileUploader } from './FileUploader';
import { Breadcrumb } from './Breadcrumb';
import { useStore } from '../../store/useStore';
import { RotateCcw } from 'lucide-react';

export function Header() {
  const reset = useStore(s => s.reset);
  const sseFileName = useStore(s => s.sseFileName);

  return (
    <header className="h-12 bg-gray-900 border-b border-gray-700 flex items-center px-4 gap-4 shrink-0">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-[10px] font-bold text-white">
          P
        </div>
        <span className="text-sm font-semibold text-white tracking-tight">Phoenix</span>
      </div>

      <div className="h-5 w-px bg-gray-700" />

      <FileUploader />

      {sseFileName && (
        <>
          <div className="h-5 w-px bg-gray-700" />
          <Breadcrumb />
        </>
      )}

      <div className="flex-1" />

      {sseFileName && (
        <button
          onClick={reset}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
      )}
    </header>
  );
}
